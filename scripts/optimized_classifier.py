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

print(f"JAX is using: {jax.devices()}")

from sklearn.metrics import roc_auc_score
from functools import partial

# =============================================================================
# 2. HUGGING FACE FEATURIZER (PyTorch - Unchanged)
# =============================================================================
class HuggingFaceFeaturizer:
    """Handles featurization of SMILES strings using PyTorch-based Hugging Face models."""
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
# 3. JAX-BASED SCALER & UTILS (Unchanged)
# =============================================================================

def scaler_init(n_features):
    """Initializes the state for the online scaler."""
    return {
        'count': jnp.zeros((), dtype=jnp.int32),
        'mean': jnp.zeros(n_features, dtype=jnp.float32),
        'M2': jnp.zeros(n_features, dtype=jnp.float32)
    }

@jax.jit
def scaler_update(scaler_state, X_batch):
    """Updates the scaler state using a batch of data."""
    batch_count = X_batch.shape[0]
    if batch_count == 0:
        return scaler_state
    batch_mean = jnp.mean(X_batch, axis=0)
    batch_M2 = jnp.sum(jnp.square(X_batch - batch_mean), axis=0)
    new_count = scaler_state['count'] + batch_count
    delta = batch_mean - scaler_state['mean']
    new_mean = scaler_state['mean'] + delta * (batch_count / new_count)
    new_M2 = scaler_state['M2'] + batch_M2 + jnp.square(delta) * scaler_state['count'] * batch_count / new_count
    return {'count': new_count, 'mean': new_mean, 'M2': new_M2}

def scaler_finalize(scaler_state):
    """Calculates final variance and scale from the accumulated M2 state."""
    count = scaler_state['count']
    mean = scaler_state['mean']
    M2 = scaler_state['M2']
    var = jnp.where(count > 0, M2 / count, 0.0)
    scale = jnp.sqrt(var)
    scale = jnp.where(scale == 0, 1.0, scale)
    return {'mean': mean, 'scale': scale}

@jax.jit
def scaler_transform(scaler_params, X):
    """Applies standard scaling with numerical stability."""
    epsilon = 1e-6
    return (X - scaler_params['mean']) / (scaler_params['scale'] + epsilon)

# =============================================================================
# 4. JAX-BASED MODEL & TRAINING CORE (Updated for Masking)
# =============================================================================

@partial(jax.jit, static_argnames=['optimizer'])
def train_step(state, batch, learning_rate, scaler_params, unique_ligands, unique_proteins, alpha, l1_ratio, optimizer):
    """
    ## MODIFIED: Now accepts and uses a mask to compute loss correctly on padded batches.
    """
    params, opt_state = state
    # Unpack the batch of data, now including the mask
    (batch_ligand_idx, batch_protein_idx), batch_y, batch_mask = batch

    def loss_fn(p):
        X_batch = jnp.concatenate([
            unique_ligands[batch_ligand_idx],
            unique_proteins[batch_protein_idx]
        ], axis=1)

        X_scaled = scaler_transform(scaler_params, X_batch)
        logits = X_scaled @ p['w'].T + p['b']

        # Calculate binary cross-entropy loss for each sample in the batch
        per_sample_loss = optax.sigmoid_binary_cross_entropy(logits.flatten(), batch_y)
        
        # **NEW**: Apply the mask to ignore losses from padded samples
        masked_loss = per_sample_loss * batch_mask
        
        # **NEW**: Average the loss only over the *real* samples in the batch
        # Use jnp.maximum to avoid division by zero if a batch is all padding
        data_loss = jnp.sum(masked_loss) / jnp.maximum(jnp.sum(batch_mask), 1)

        l2_penalty = 0.5 * (1 - l1_ratio) * alpha * jnp.sum(p['w']**2)
        
        return data_loss + l2_penalty

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    l1_shrinkage = learning_rate * alpha * l1_ratio
    params['w'] = jnp.sign(params['w']) * jnp.maximum(0, jnp.abs(params['w']) - l1_shrinkage)

    return (params, opt_state), loss

@partial(jax.jit, static_argnames=['batch_size'])
def predict_probas_batched(params, scaler_params, ligand_indices, protein_indices, unique_ligands, unique_proteins, batch_size):
    """(Unchanged) JIT-compiled batched inference."""
    original_num_samples = ligand_indices.shape[0]
    padding = (batch_size - (original_num_samples % batch_size)) % batch_size
    ligand_indices_padded = jnp.pad(ligand_indices, (0, padding))
    protein_indices_padded = jnp.pad(protein_indices, (0, padding))

    num_samples_padded = original_num_samples + padding
    num_batches = num_samples_padded // batch_size

    def body_fn(i, preds_array):
        start_idx = i * batch_size
        batch_ligand_indices = jax.lax.dynamic_slice_in_dim(ligand_indices_padded, start_idx, batch_size, axis=0)
        batch_protein_indices = jax.lax.dynamic_slice_in_dim(protein_indices_padded, start_idx, batch_size, axis=0)
        
        X_batch = jnp.concatenate([
            unique_ligands[batch_ligand_indices],
            unique_proteins[batch_protein_indices]
        ], axis=1)
        
        X_scaled = scaler_transform(scaler_params, X_batch)
        logits = X_scaled @ params['w'].T + params['b']
        probas = jax.nn.sigmoid(logits)
        
        return jax.lax.dynamic_update_slice(preds_array, probas.flatten(), [start_idx])

    initial_preds = jnp.zeros(num_samples_padded)
    final_preds_padded = jax.lax.fori_loop(0, num_batches, body_fn, initial_preds)
    final_preds = jax.lax.slice(final_preds_padded, [0], [original_num_samples])
    
    return final_preds

# =============================================================================
# 5. MAIN SCRIPT LOGIC (Updated for Masking)
# =============================================================================

def main(args):
    start_time = time.time()

    # --- Phase 0: Load and Prepare All Data In-Memory (Unchanged) ---
    print("--- Loading and preparing data in-memory ---")
    train_df = pd.read_parquet(args.train_data)
    train_df['target_id'] = train_df['uniprot_id'] + '_' + train_df['mutation']
    protein_archive = np.load(args.protein_features_path)
    available_targets = set(protein_archive.files)
    train_df = train_df[train_df['target_id'].isin(available_targets)].reset_index(drop=True)
    test_df = None
    if args.test_data:
        test_df = pd.read_parquet(args.test_data)
        test_df['target_id'] = test_df['uniprot_id'] + '_' + test_df['mutation']
        test_df = test_df[test_df['target_id'].isin(available_targets)].reset_index(drop=True)
    all_smiles = pd.concat([train_df['Ligand SMILES'], test_df['Ligand SMILES'] if test_df is not None else pd.Series()]).unique()
    all_proteins = pd.concat([train_df['target_id'], test_df['target_id'] if test_df is not None else pd.Series()]).unique()
    all_smiles.sort()
    all_proteins.sort()
    global_smiles_map = {smile: i for i, smile in enumerate(all_smiles)}
    global_protein_map = {pid: i for i, pid in enumerate(all_proteins)}
    print("Featurizing all unique SMILES...")
    hf_featurizer = HuggingFaceFeaturizer()
    smiles_embeddings = hf_featurizer.featurize(all_smiles, args.ligand_featurizer, batch_size=args.batch_size)
    protein_embeddings = np.vstack([protein_archive[pid] for pid in all_proteins])
    train_ligand_indices = np.array([global_smiles_map[s] for s in train_df['Ligand SMILES']], dtype=np.int32)
    train_protein_indices = np.array([global_protein_map[p] for p in train_df['target_id']], dtype=np.int32)
    y_train = train_df['confers_resistance'].values.astype(np.int32)
    if test_df is not None:
        test_ligand_indices = np.array([global_smiles_map[s] for s in test_df['Ligand SMILES']], dtype=np.int32)
        test_protein_indices = np.array([global_protein_map[p] for p in test_df['target_id']], dtype=np.int32)
        y_test = test_df['confers_resistance'].values.astype(np.int32)
    print("✅ In-memory data preparation complete.")

    # --- JAX Setup (Unchanged) ---
    key = jax.random.PRNGKey(args.random_seed)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate=args.learning_rate)
    )

    # --- Phase 1: Load all data to JAX device (Unchanged) ---
    print("\n--- Loading all data to JAX device ---")
    unique_ligands = jax.device_put(smiles_embeddings)
    unique_proteins = jax.device_put(protein_embeddings)
    train_ligand_indices = jax.device_put(train_ligand_indices)
    train_protein_indices = jax.device_put(train_protein_indices)
    train_labels = jax.device_put(y_train)
    print("✅ Training data loaded to JAX device.")

    # --- Phase 1.5: Fit a Standard Scaler (Unchanged) ---
    print("\n--- Fitting a Standard scaler on-device ---")
    n_features = unique_ligands.shape[1] + unique_proteins.shape[1]
    n_train_samples = train_labels.shape[0]
    fit_batch_size = args.fit_batch_size
    @jax.jit
    def fit_standard_scaler_batch(scaler_state, batch_ligand_idx, batch_protein_idx, unique_ligands, unique_proteins):
        X_batch = jnp.concatenate([unique_ligands[batch_ligand_idx], unique_proteins[batch_protein_idx]], axis=1)
        return scaler_update(scaler_state, X_batch)
    scaler_state = scaler_init(n_features)
    for i in tqdm(range(0, n_train_samples, fit_batch_size), desc="Standard scaler"):
        start_idx, end_idx = i, min(i + fit_batch_size, n_train_samples)
        batch_ligand_idx = train_ligand_indices[start_idx:end_idx]
        batch_protein_idx = train_protein_indices[start_idx:end_idx]
        scaler_state = fit_standard_scaler_batch(
            scaler_state, batch_ligand_idx, batch_protein_idx, unique_ligands, unique_proteins
        )
    scaler_params = scaler_finalize(scaler_state)
    scaler_params['mean'].block_until_ready()
    print("✅ Standard scaler fitted on-device.")

    # --- Phase 3: Initialize Model State (Unchanged) ---
    params = {'w': jnp.zeros((1, n_features)), 'b': jnp.zeros(1)}
    opt_state = optimizer.init(params)
    state = (params, opt_state)

    # --- Phase 4: The JAX Training Loop (MODIFIED FOR MASKING) ---
    print(f"\n--- Training Model on JAX ---")

    # **NEW**: Create a mask for the training data.
    train_batch_size = args.train_batch_size
    padding = (train_batch_size - (n_train_samples % train_batch_size)) % train_batch_size
    n_train_padded = n_train_samples + padding
    num_batches = n_train_padded // train_batch_size

    # Pad all training arrays and also create a corresponding mask.
    train_ligand_indices_padded = jnp.pad(train_ligand_indices, (0, padding), constant_values=0)
    train_protein_indices_padded = jnp.pad(train_protein_indices, (0, padding), constant_values=0)
    train_labels_padded = jnp.pad(train_labels, (0, padding), constant_values=0)
    
    # The mask is 1 for real data and 0 for padding.
    train_mask = jnp.ones_like(train_labels, dtype=jnp.float32)
    train_mask_padded = jnp.pad(train_mask, (0, padding), constant_values=0)


    def epoch_loop_body(state, batch_indices, lr, scaler_params,
                        unique_ligands, unique_proteins, alpha, l1_ratio, optimizer,
                        # Pass padded arrays to the loop body
                        train_ligand_indices_padded, train_protein_indices_padded, 
                        train_labels_padded, train_mask_padded): # Pass the mask
        # Gather the actual data for the batch, now including the mask
        batch_data = (
            (train_ligand_indices_padded[batch_indices], train_protein_indices_padded[batch_indices]),
            train_labels_padded[batch_indices],
            train_mask_padded[batch_indices] # Pass the mask slice for this batch
        )
        new_state, loss = train_step(state, batch_data, lr, scaler_params, unique_ligands, unique_proteins, alpha, l1_ratio, optimizer)
        return new_state, loss

    loop_body_partial = partial(epoch_loop_body,
                                lr=args.learning_rate,
                                scaler_params=scaler_params,
                                unique_ligands=unique_ligands,
                                unique_proteins=unique_proteins,
                                alpha=args.alpha, l1_ratio=args.l1_ratio,
                                optimizer=optimizer,
                                # Pass the newly created padded arrays
                                train_ligand_indices_padded=train_ligand_indices_padded,
                                train_protein_indices_padded=train_protein_indices_padded,
                                train_labels_padded=train_labels_padded,
                                train_mask_padded=train_mask_padded) # Pass the full mask

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        key, subkey = jax.random.split(key)
        
        perm = jax.random.permutation(subkey, n_train_padded)
        perm_batched = perm.reshape((num_batches, train_batch_size))

        state, losses = jax.lax.scan(loop_body_partial, state, perm_batched)
        
        losses[-1].block_until_ready()

        duration = time.time() - epoch_start_time
        it_per_sec = n_train_samples / duration
        print(f"Epoch {epoch + 1}/{args.epochs} complete in {duration:.2f}s (Avg loss: {losses.mean():.4f}). Effective speed: {it_per_sec:,.0f} it/s")

    # --- Phase 5: Evaluation (Unchanged) ---
    print("\n--- Evaluating Model ---")
    final_params, _ = state 

    predictions = predict_probas_batched(
        final_params, scaler_params,
        train_ligand_indices, train_protein_indices,
        unique_ligands, unique_proteins,
        batch_size=args.predict_batch_size
    )
    predictions.block_until_ready()
    score = roc_auc_score(y_train, np.asarray(predictions))
    print("\n--- Results ---")
    print(f"Train Set ROC AUC Score: {score:.4f}")

    if test_df is not None:
        test_ligand_indices_jax = jax.device_put(test_ligand_indices)
        test_protein_indices_jax = jax.device_put(test_protein_indices)
        predictions = predict_probas_batched(
            final_params, scaler_params,
            test_ligand_indices_jax, test_protein_indices_jax,
            unique_ligands, unique_proteins,
            batch_size=args.predict_batch_size
        )
        predictions.block_until_ready()
        score = roc_auc_score(y_test, np.asarray(predictions))
        print(f"Test Set ROC AUC Score: {score:.4f}")

    num_protein_features = protein_embeddings.shape[1]
    selected_coeffs = final_params['w'][0, -num_protein_features:]
    num_selected = np.sum(selected_coeffs != 0)
    print(f"\nL1 regularization selected {num_selected} out of {num_protein_features} genomic features.")
    total_time = time.time() - start_time
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ultimate Performance Out-of-Core Classifier with JAX.")
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--protein_features_path', type=str, required=True)
    parser.add_argument('--ligand_featurizer', type=str, default='ibm/MoLFormer-XL-both-10pct')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fit_batch_size', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--predict_batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--l1_ratio', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
