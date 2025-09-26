import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import time
import json
import shutil

# PyTorch imports are still needed for the featurizer
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

# JAX and Optax imports for the new training core
import jax
import jax.numpy as jnp
import optax

print(jax.devices())


from sklearn.metrics import roc_auc_score

from functools import partial

# =============================================================================
# 1. DATA PREPARATION (Largely Unchanged)
# =============================================================================
# prepare_shards remains the same
def prepare_shards(df, smiles_map, protein_archive_path, shard_dir, df_type='train'):
    print(f"--- Preparing {df_type} shards and global feature stores in {shard_dir} ---")
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    
    protein_archive = np.load(protein_archive_path)
    
    unique_smiles = df['Ligand SMILES'].unique()
    unique_proteins = df['target_id'].unique()
    
    smiles_to_global_idx = {smile: i for i, smile in enumerate(unique_smiles)}
    protein_to_global_idx = {pid: i for i, pid in enumerate(unique_proteins)}

    np.save(os.path.join(shard_dir, 'unique_ligands.npy'), np.vstack([smiles_map[s] for s in unique_smiles]))
    np.save(os.path.join(shard_dir, 'unique_proteins.npy'), np.vstack([protein_archive[pid] for pid in unique_proteins]))
    
    # Save the indices
    ligand_indices = np.array([smiles_to_global_idx[s] for s in df['Ligand SMILES']], dtype=np.int32)
    protein_indices = np.array([protein_to_global_idx[p] for p in df['target_id']], dtype=np.int32)
    np.save(os.path.join(shard_dir, 'df_ligand_indices.npy'), ligand_indices)
    np.save(os.path.join(shard_dir, 'df_protein_indices.npy'), protein_indices)
    np.save(os.path.join(shard_dir, 'df_labels.npy'), df['is_effective_against_mutant'].values.astype(np.int32))
        
    print(f"✅ {df_type.capitalize()} cache preparation complete.")

# =============================================================================
# 2. HUGGING FACE FEATURIZER (PyTorch - Unchanged)
# =============================================================================
class HuggingFaceFeaturizer:
    def __init__(self):
        self.models_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"HuggingFaceFeaturizer using device: {self.device}")

    def _load_model(self, model_name):
        if model_name not in self.models_cache:
            ModelClass = T5EncoderModel if 't5' in model_name else AutoModel
            TokenizerClass = T5Tokenizer if 't5' in model_name else AutoTokenizer
            model = ModelClass.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
            tokenizer = TokenizerClass.from_pretrained(model_name, trust_remote_code=True)
            self.models_cache[model_name] = {'model': model, 'tokenizer': tokenizer}
        return self.models_cache[model_name]

    def featurize(self, sequences, model_name, batch_size=32):
        model_dict = self._load_model(model_name)
        model, tokenizer = model_dict['model'], model_dict['tokenizer']
        all_embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Featurizing with {os.path.basename(model_name)}"):
            batch_seqs = list(sequences[i:i + batch_size])
            inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0).numpy()

# =============================================================================
# 3. JAX-BASED CLASSIFIER AND TRAINING LOOP
# =============================================================================

@partial(jax.jit, static_argnames=['optimizer'])
def train_step(state, data, unique_ligands, unique_proteins, optimizer, alpha, l1_ratio, t0, samples_seen):
    """A JIT-compiled function for a single training step."""
    params, opt_state = state
    (ligand_idx, protein_idx), y = data

    lr = 1.0 / (alpha * (t0 + samples_seen))

    def loss_fn(p):
        ligand_vec = unique_ligands[ligand_idx]
        protein_vec = unique_proteins[protein_idx]
        X_sample = jnp.concatenate([ligand_vec, protein_vec])
        logits = X_sample @ p['w'].T + p['b']
        loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
        l2_penalty = 0.5 * (1 - l1_ratio) * alpha * jnp.sum(p['w']**2)
        return loss + l2_penalty

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    w = params['w']
    l1_shrinkage = lr * alpha * l1_ratio
    params['w'] = jnp.sign(w) * jnp.maximum(0, jnp.abs(w) - l1_shrinkage)
    
    return (params, opt_state), loss

@partial(jax.jit, static_argnames=['batch_size'])
def predict_probas_batched(params, ligand_indices, protein_indices, unique_ligands, unique_proteins, batch_size):
    """
    JIT-compiled batched inference that assembles features on-the-fly and uses
    JIT-compatible dynamic slicing to prevent tracer errors.
    """
    num_samples = ligand_indices.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    def body_fn(i, preds_array):
        start_idx = i * batch_size
        
        batch_ligand_indices = jax.lax.dynamic_slice_in_dim(ligand_indices, start_idx, batch_size, axis=0)
        batch_protein_indices = jax.lax.dynamic_slice_in_dim(protein_indices, start_idx, batch_size, axis=0)
        
        X_batch = jnp.concatenate([
            unique_ligands[batch_ligand_indices],
            unique_proteins[batch_protein_indices]
        ], axis=1)
        
        logits = X_batch @ params['w'].T + params['b']
        probas = jax.nn.sigmoid(logits)

        # --- THIS IS THE CRITICAL FIX ---
        # Replace the NumPy-style slice assignment with JAX's explicit dynamic update function.
        # It takes the full array, the values to update with, and the starting index.
        return jax.lax.dynamic_update_slice(preds_array, probas.flatten(), [start_idx])

    initial_preds = jnp.zeros(num_samples)
    final_preds = jax.lax.fori_loop(0, num_batches, body_fn, initial_preds)
    return final_preds

# =============================================================================
# 4. MAIN SCRIPT LOGIC
# =============================================================================

def main(args):
    start_time = time.time()
    
    # --- Phase 0: Data Loading and Shard Caching (Corrected) ---
    print("--- Loading and filtering dataframes ---")
    train_df = pd.read_parquet(args.train_data)
    train_df['target_id'] = train_df['uniprot_id'] + '_' + train_df['mutation']
    
    with np.load(args.protein_features_path) as archive:
        available_targets = set(archive.files)
    train_df = train_df[train_df['target_id'].isin(available_targets)].reset_index(drop=True)
    
    if args.test_data:
        test_df = pd.read_parquet(args.test_data)
        test_df['target_id'] = test_df['uniprot_id'] + '_' + test_df['mutation']
        test_df = test_df[test_df['target_id'].isin(available_targets)].reset_index(drop=True)

    if args.shard_dir is None:
        base_dir = os.path.dirname(args.train_data)
        args.shard_dir = os.path.join(base_dir, "shard_cache")
    
    train_shard_dir = os.path.join(args.shard_dir, 'train')
    canary_file = os.path.join(train_shard_dir, 'unique_ligands.npy')
    regen_needed = not os.path.exists(canary_file) or \
                   os.path.getmtime(args.train_data) > os.path.getmtime(canary_file) or \
                   os.path.getmtime(args.protein_features_path) > os.path.getmtime(canary_file)

    if regen_needed:
        print("Train shard cache not found or is stale. Regenerating...")
        hf_featurizer = HuggingFaceFeaturizer()
        # Create a combined set of unique smiles from train and test if applicable
        if args.test_data:
            unique_smiles_all = pd.concat([train_df['Ligand SMILES'], test_df['Ligand SMILES']]).unique()
        else:
            unique_smiles_all = train_df['Ligand SMILES'].unique()
        
        smiles_map = {s: emb for s, emb in zip(unique_smiles_all, hf_featurizer.featurize(unique_smiles_all, args.ligand_featurizer, batch_size=args.batch_size))}
        prepare_shards(train_df, smiles_map, args.protein_features_path, train_shard_dir, 'train')
    else:
        print(f"✅ Using existing train shard cache at {train_shard_dir}")

    if args.test_data:
        test_shard_dir = os.path.join(args.shard_dir, 'test')
        canary_file = os.path.join(test_shard_dir, 'unique_ligands.npy')
        regen_needed_test = not os.path.exists(canary_file) or \
                            os.path.getmtime(args.test_data) > os.path.getmtime(canary_file) or \
                            os.path.getmtime(args.protein_features_path) > os.path.getmtime(canary_file)
        
        if regen_needed_test:
            print("Test shard cache not found or is stale. Regenerating...")
            if 'smiles_map' not in locals(): # In case train cache was not regenerated
                 hf_featurizer = HuggingFaceFeaturizer()
                 unique_smiles_all = pd.concat([train_df['Ligand SMILES'], test_df['Ligand SMILES']]).unique()
                 smiles_map = {s: emb for s, emb in zip(unique_smiles_all, hf_featurizer.featurize(unique_smiles_all, args.ligand_featurizer, batch_size=args.batch_size))}
            prepare_shards(test_df, smiles_map, args.protein_features_path, test_shard_dir, 'test')
        else:
            print(f"✅ Using existing test shard cache at {test_shard_dir}")

    # JAX Setup
    key = jax.random.PRNGKey(args.random_seed)
    optimizer = optax.sgd(learning_rate=1.0) 

    # --- Phase 1: Load all data to JAX device (GPU) ---
    print("\n--- Loading all data to JAX device ---")
    unique_ligands = jax.device_put(np.load(os.path.join(train_shard_dir, 'unique_ligands.npy')))
    unique_proteins = jax.device_put(np.load(os.path.join(train_shard_dir, 'unique_proteins.npy')))
    train_ligand_indices = jax.device_put(np.load(os.path.join(train_shard_dir, 'df_ligand_indices.npy')))
    train_protein_indices = jax.device_put(np.load(os.path.join(train_shard_dir, 'df_protein_indices.npy')))
    train_labels = jax.device_put(np.load(os.path.join(train_shard_dir, 'df_labels.npy')))
    print("✅ Training data loaded to JAX device.")

    # --- Phase 2: Heuristic for `t0` (data-independent) ---
    print("--- Estimating t0 using scikit-learn's exact heuristic ---")
    typw = np.sqrt(1.0 / np.sqrt(args.alpha))
    dloss_probe = typw + 1.0
    eta0 = typw / dloss_probe
    t0 = 1.0 / (eta0 * args.alpha)
    print(f"✅ Heuristic found eta0={eta0:.4f}, calculated t0={t0:.2f}")

    # --- Phase 3: Initialize Model State ---
    n_features = unique_ligands.shape[1] + unique_proteins.shape[1]
    params = {'w': jnp.zeros((1, n_features)), 'b': jnp.zeros(1)}
    opt_state = optimizer.init(params)
    state = (params, opt_state)

    # --- Phase 4: The JAX Training Loop ---
    print(f"\n--- Phase 4: Training Model on JAX ---")
    samples_seen = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, train_labels.shape[0])

        def epoch_loop_body(i, current_state_and_loss):
            current_state, _ = current_state_and_loss
            idx = perm[i]
            data = ((train_ligand_indices[idx], train_protein_indices[idx]), train_labels[idx])
            current_samples_seen = samples_seen + i
            new_state, loss = train_step(
                current_state, data, unique_ligands, unique_proteins, 
                optimizer, args.alpha, args.l1_ratio, t0, current_samples_seen
            )
            return new_state, loss

        # JIT-compile and run the entire loop for the epoch
        (final_state_in_epoch, final_loss), _ = jax.lax.scan(
            lambda carry, i: (epoch_loop_body(i, carry), None), 
            (state, 0.0), 
            jnp.arange(train_labels.shape[0])
        )
        
        # --- THIS IS THE CRITICAL FIX FOR ACCURATE TIMING ---
        # Force the CPU to wait for the GPU to finish all work for this epoch.
        final_loss.block_until_ready()
        state = final_state_in_epoch # Update the state for the next epoch
        
        samples_seen += train_labels.shape[0]

        duration = time.time() - epoch_start_time
        it_per_sec = train_labels.shape[0] / duration
        print(f"Epoch {epoch + 1}/{args.epochs} complete in {duration:.2f}s. Effective speed: {it_per_sec:,.0f} it/s (actual GPU time)")

    # --- Phase 5: Evaluation ---
    if args.test_data:
        print("\n--- Phase 5: Evaluating Model ---")
        # Load test indices and labels
        test_ligand_indices = jax.device_put(np.load(os.path.join(test_shard_dir, 'df_ligand_indices.npy')))
        test_protein_indices = jax.device_put(np.load(os.path.join(test_shard_dir, 'df_protein_indices.npy')))
        y_test = np.load(os.path.join(test_shard_dir, 'df_labels.npy'))

        # Get the final trained parameters from the state
        final_params, _ = state
        
        # --- Call the new batched prediction function ---
        # We pass the indices and embedding tables, NOT a pre-assembled X_test
        predictions = predict_probas_batched(
            final_params, 
            test_ligand_indices, test_protein_indices,
            unique_ligands, unique_proteins, # Use the embeddings already on the GPU
            batch_size=args.predict_batch_size # e.g., 4096
        )
        
        # Block until all GPU prediction work is complete
        predictions.block_until_ready()

        score = roc_auc_score(y_test, np.asarray(predictions))
        print("\n--- Results ---")
        print(f"Test Set ROC AUC Score: {score:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ultimate Performance Out-of-Core Classifier.")
    # Path args
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--protein_features_path', type=str, required=True, help="Path to the source .npz file.")
    parser.add_argument('--shard_dir', type=str, default=None, help="Directory to cache data shards. Defaults to a subdir next to the train_data.")
    parser.add_argument('--ligand_featurizer', type=str, default='ibm/MoLFormer-XL-both-10pct')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--predict_batch_size', type=int, default=32)

    # Model args
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16'], default='float32')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--alpha', type=float, default=1e-4, help="Regularization strength")
    parser.add_argument('--l1_ratio', type=float, default=0.15, help="Elastic Net mixing parameter")
    
    # Training loop args
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_shards', type=int, default=64, help="Number of shards to create for the cache.")
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
