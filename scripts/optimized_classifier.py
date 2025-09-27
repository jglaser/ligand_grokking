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
# 1. DATA PREPARATION (Corrected for Determinism and Correctness)
# =============================================================================
def prepare_shards(df, global_smiles_map, global_protein_map, shard_dir, df_type='train'):
    """
    Creates index and label files for a data split (train/test) by referencing
    a global, sorted mapping of features.
    """
    print(f"--- Preparing {df_type} indices in {shard_dir} ---")
    os.makedirs(shard_dir, exist_ok=True)
    
    # Use the global maps to create indices for the current dataframe split
    ligand_indices = np.array([global_smiles_map[s] for s in df['Ligand SMILES']], dtype=np.int32)
    protein_indices = np.array([global_protein_map[p] for p in df['target_id']], dtype=np.int32)
    
    # Save the mapping and labels specific to this split
    np.save(os.path.join(shard_dir, 'df_ligand_indices.npy'), ligand_indices)
    np.save(os.path.join(shard_dir, 'df_protein_indices.npy'), protein_indices)
    np.save(os.path.join(shard_dir, 'df_labels.npy'), df['confers_resistance'].values.astype(np.int32))
    print(f"✅ {df_type.capitalize()} index preparation complete.")

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
# 3. JAX-BASED STANDARD SCALER (NEW)
# =============================================================================
def scaler_init(n_features):
    """Initializes the state for the online scaler."""
    # State consists of count, mean, and M2 (sum of squares of differences from the current mean)
    return {
        'count': jnp.zeros((), dtype=jnp.int32),
        'mean': jnp.zeros(n_features, dtype=jnp.float32),
        'M2': jnp.zeros(n_features, dtype=jnp.float32)
    }

@jax.jit
def scaler_update(scaler_state, X_batch):
    """
    Updates the scaler state using a batch of data. Implements a vectorized
    version of Welford's algorithm for combining summary statistics.
    """
    batch_count = X_batch.shape[0]
    # This check is important for the last batch which might be empty if total_size % batch_size == 0
    if batch_count == 0:
        return scaler_state
    
    batch_mean = jnp.mean(X_batch, axis=0)
    batch_M2 = jnp.sum(jnp.square(X_batch - batch_mean), axis=0)
    
    # Combine the statistics from the existing state and the new batch
    new_count = scaler_state['count'] + batch_count
    delta = batch_mean - scaler_state['mean']
    
    new_mean = scaler_state['mean'] + delta * (batch_count / new_count)
    
    # M2 update: old M2 + batch M2 + correction term for combining variances
    new_M2 = scaler_state['M2'] + batch_M2 + jnp.square(delta) * scaler_state['count'] * batch_count / new_count
    
    return {'count': new_count, 'mean': new_mean, 'M2': new_M2}

def scaler_finalize(scaler_state):
    """Calculates final variance and scale from the accumulated M2 state."""
    count = scaler_state['count']
    mean = scaler_state['mean']
    M2 = scaler_state['M2']
    
    # Calculate population variance. Use a safe default of 0 if no data was seen.
    var = jnp.where(count > 0, M2 / count, 0.0)
    scale = jnp.sqrt(var)
    # Don't scale features with zero variance to avoid NaNs.
    scale = jnp.where(scale == 0, 1.0, scale) 
    
    return {'mean': mean, 'scale': scale}

@jax.jit
def scaler_transform(scaler_params, X):
    """Applies MaxAbs scaling with numerical stability."""
    epsilon = 1e-6
    return X / (scaler_params['max_abs'] + epsilon)

def get_quantiles_from_hist(histograms, bin_edges_matrix, quantiles):
    """Calculates quantiles from histograms with per-feature bin edges."""
    cdf = jnp.cumsum(histograms, axis=-1)
    total_counts = cdf[..., -1]
    
    target_counts = total_counts[..., None] * quantiles[None, :] - 1e-6
    
    # Define a function for a single row to vmap
    def find_indices(single_cdf, single_target_counts):
        return jnp.searchsorted(single_cdf, single_target_counts)
        
    quantile_indices = jax.vmap(find_indices)(cdf, target_counts)
    
    # Interpolate to find the quantile value within the correct bins for each feature
    bin_widths = (bin_edges_matrix[:, 1:] - bin_edges_matrix[:, :-1])
    
    # Gather the specific bin widths and starting edges for each quantile index
    # This is an advanced indexing operation
    relevant_bin_starts = jax.vmap(lambda x, y: x[y])(bin_edges_matrix, quantile_indices)
    relevant_bin_widths = jax.vmap(lambda x, y: x[y])(bin_widths, quantile_indices)

    return relevant_bin_starts + relevant_bin_widths / 2


@jax.jit
def get_columnwise_min_max_batch(batch_ligand_idx, batch_protein_idx, unique_ligands, unique_proteins):
    """JIT-compiled function to get the column-wise min and max of a feature batch."""
    X_batch = jnp.concatenate([
        unique_ligands[batch_ligand_idx],
        unique_proteins[batch_protein_idx]
    ], axis=1)
    return jnp.min(X_batch, axis=0), jnp.max(X_batch, axis=0)


@partial(jax.jit, static_argnames=['feature_chunk_size'])
def fit_robust_scaler_batch(batch_ligand_idx, batch_protein_idx, unique_ligands, unique_proteins, 
                            bin_edges_chunk, feature_start_idx, feature_chunk_size):
    """
    JIT-compiled function to assemble one batch, slice a chunk of features, 
    and calculate histograms using per-feature bin edges.
    """
    X_batch = jnp.concatenate([
        unique_ligands[batch_ligand_idx],
        unique_proteins[batch_protein_idx]
    ], axis=1)
    
    X_batch_chunk = jax.lax.dynamic_slice_in_dim(X_batch, feature_start_idx, feature_chunk_size, axis=1)
    
    # vmap now iterates over feature columns (axis=1) and corresponding bin_edges for that feature
    return jax.vmap(lambda x, bins: jnp.histogram(x, bins=bins)[0], in_axes=(1, 0))(X_batch_chunk, bin_edges_chunk)

# =============================================================================
# 4. JAX-BASED CLASSIFIER AND TRAINING LOOP (Updated)
# =============================================================================
@jax.jit
def train_step(state, data, lr, scaler_params, unique_ligands, unique_proteins, alpha, l1_ratio):
    """A JIT-compiled function for a single training step, now including scaling."""
    params, opt_state = state
    (ligand_idx, protein_idx), y = data

    def loss_fn(p):
        ligand_vec = unique_ligands[ligand_idx]
        protein_vec = unique_proteins[protein_idx]
        X_sample = jnp.concatenate([ligand_vec, protein_vec])
        
        # --- Apply scaling on-the-fly ---
        X_scaled = scaler_transform(scaler_params, X_sample)
        
        logits = X_scaled @ p['w'].T + p['b']

        loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
        l2_penalty = 0.5 * (1 - l1_ratio) * alpha * jnp.sum(p['w']**2)
        return loss + l2_penalty

    loss, grads = jax.value_and_grad(loss_fn)(params)

#    # 2. Calculate the global norm of the gradient tree
#    grad_norm = optax.global_norm(grads)
#    clip_threshold = 1.0 # A standard, robust default value
#
#    # 3. Conditionally clip the gradients if their norm exceeds the threshold
#    # This prevents any single update from becoming explosively large.
#    trigger = grad_norm > clip_threshold
#    clipped_grads = jax.tree_util.tree_map(
#        lambda g: jax.lax.select(trigger, g * clip_threshold / grad_norm, g), grads
#    )
#
#    updates = jax.tree_util.tree_map(lambda g: -lr * g, clipped_grads)
    updates = jax.tree_util.tree_map(lambda g: -lr * g, grads)
    params = optax.apply_updates(params, updates)

    # Apply L1 penalty (proximal operator for elastic net)
    w = params['w']
    l1_shrinkage = lr * alpha * l1_ratio
    params['w'] = jnp.sign(w) * jnp.maximum(0, jnp.abs(w) - l1_shrinkage)

    return (params, opt_state), loss

@partial(jax.jit, static_argnames=['batch_size'])
def predict_probas_batched(params, scaler_params, ligand_indices, protein_indices, unique_ligands, unique_proteins, batch_size):
    """
    JIT-compiled batched inference. This version pads the input data to ensure all
    batches have a static, uniform size, respecting JAX's API constraints.
    """
    # <<< FIX: The entire function logic is updated to use padding.
    
    # 1. Get the original number of samples
    original_num_samples = ligand_indices.shape[0]

    # 2. Calculate the required padding to make the total a multiple of the batch size
    padding = (batch_size - (original_num_samples % batch_size)) % batch_size
    
    # 3. Pad the input index arrays to the new length.
    ligand_indices_padded = jnp.pad(ligand_indices, (0, padding))
    protein_indices_padded = jnp.pad(protein_indices, (0, padding))

    # 4. Calculate the new total size and number of batches for the loop
    num_samples_padded = original_num_samples + padding
    num_batches = num_samples_padded // batch_size # Use integer division

    def body_fn(i, preds_array):
        start_idx = i * batch_size
        
        # Now, the slice_size is always the static 'batch_size', which JAX requires.
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

    # 5. Initialize the predictions array with the padded size
    initial_preds = jnp.zeros(num_samples_padded)
    final_preds_padded = jax.lax.fori_loop(0, num_batches, body_fn, initial_preds)
    
    # 6. CRITICAL: Slice the padded results back to the original number of samples before returning
    final_preds = jax.lax.slice(final_preds_padded, [0], [original_num_samples])
    
    return final_preds

# =============================================================================
# 5. MAIN SCRIPT LOGIC (Updated & Corrected)
# =============================================================================

def main(args):
    start_time = time.time()
    
    # --- Phase 0: Data Loading and Shard Caching (Corrected) ---
    print("--- Loading and filtering dataframes ---")
    train_df = pd.read_parquet(args.train_data)
    train_df['target_id'] = train_df['uniprot_id'] + '_' + train_df['mutation']
    
    protein_archive = np.load(args.protein_features_path)
    available_targets = set(protein_archive.files)
    train_df = train_df[train_df['target_id'].isin(available_targets)].reset_index(drop=True)
    
    if args.test_data:
        test_df = pd.read_parquet(args.test_data)
        test_df['target_id'] = test_df['uniprot_id'] + '_' + test_df['mutation']
        test_df = test_df[test_df['target_id'].isin(available_targets)].reset_index(drop=True)

    if args.shard_dir is None:
        base_dir = os.path.dirname(args.train_data)
        args.shard_dir = os.path.join(base_dir, "shard_cache")
    
    # --- This canary file now checks the global feature store ---
    canary_file = os.path.join(args.shard_dir, 'unique_ligands.npy')
    regen_needed = not os.path.exists(canary_file) or \
                   os.path.getmtime(args.train_data) > os.path.getmtime(canary_file)

    if regen_needed:
        print("Global feature cache not found or is stale. Regenerating...")
        if os.path.exists(args.shard_dir):
            shutil.rmtree(args.shard_dir)
        os.makedirs(args.shard_dir, exist_ok=True)
        
        # 1. Create unified, sorted lists of all unique SMILES and proteins
        all_smiles = pd.concat([train_df['Ligand SMILES'], test_df['Ligand SMILES'] if args.test_data else pd.Series()]).unique()
        all_proteins = pd.concat([train_df['target_id'], test_df['target_id'] if args.test_data else pd.Series()]).unique()
        
        # --- SORTING IS THE KEY TO DETERMINISM ---
        all_smiles.sort()
        all_proteins.sort()
        
        # 2. Featurize SMILES and create global maps
        hf_featurizer = HuggingFaceFeaturizer()
        smiles_embeddings = hf_featurizer.featurize(all_smiles, args.ligand_featurizer, batch_size=args.batch_size)
        
        global_smiles_map = {smile: i for i, smile in enumerate(all_smiles)}
        global_protein_map = {pid: i for i, pid in enumerate(all_proteins)}

        # 3. Save the single, global, sorted feature tables
        np.save(os.path.join(args.shard_dir, 'unique_ligands.npy'), smiles_embeddings)
        np.save(os.path.join(args.shard_dir, 'unique_proteins.npy'), np.vstack([protein_archive[pid] for pid in all_proteins]))
        print("✅ Global feature tables created.")

        # 4. Create index files for each data split using the global maps
        train_shard_dir = os.path.join(args.shard_dir, 'train')
        prepare_shards(train_df, global_smiles_map, global_protein_map, train_shard_dir, 'train')
        
        if args.test_data:
            test_shard_dir = os.path.join(args.shard_dir, 'test')
            prepare_shards(test_df, global_smiles_map, global_protein_map, test_shard_dir, 'test')
    else:
        print(f"✅ Using existing data shard cache at {args.shard_dir}")

    # --- JAX Setup ---
    key = jax.random.PRNGKey(args.random_seed)
    optimizer = optax.sgd(learning_rate=args.learning_rate)

    # --- Phase 1: Load GLOBAL data and TRAIN indices to JAX device ---
    print("\n--- Loading all data to JAX device ---")
    train_shard_dir = os.path.join(args.shard_dir, 'train')
    unique_ligands = jax.device_put(np.load(os.path.join(args.shard_dir, 'unique_ligands.npy')))
    unique_proteins = jax.device_put(np.load(os.path.join(args.shard_dir, 'unique_proteins.npy')))
    
    train_ligand_indices = jax.device_put(np.load(os.path.join(train_shard_dir, 'df_ligand_indices.npy')))
    train_protein_indices = jax.device_put(np.load(os.path.join(train_shard_dir, 'df_protein_indices.npy')))
    train_labels = jax.device_put(np.load(os.path.join(train_shard_dir, 'df_labels.npy')))
    print("✅ Training data loaded to JAX device.")

    # --- Phase 1.5: Fit a MaxAbs Scaler (Definitive Version) ---
    print("\n--- Fitting a MaxAbs Scaler on-device ---")
    n_features = unique_ligands.shape[1] + unique_proteins.shape[1]
    n_train_samples = train_labels.shape[0]
    fit_batch_size = args.fit_batch_size

    # --- Find the per-feature maximum absolute value ---
    max_abs_vector = jnp.zeros((n_features,))

    @jax.jit
    def get_columnwise_max_abs_batch(batch_ligand_idx, batch_protein_idx, unique_ligands, unique_proteins):
        """JIT-compiled function to get the column-wise max_abs of a feature batch."""
        X_batch = jnp.concatenate([
            unique_ligands[batch_ligand_idx],
            unique_proteins[batch_protein_idx]
        ], axis=1)
        return jnp.max(jnp.abs(X_batch), axis=0)

    for i in tqdm(range(0, n_train_samples, fit_batch_size), desc="Finding Max Abs"):
        start_idx, end_idx = i, min(i + fit_batch_size, n_train_samples)
        batch_ligand_idx = train_ligand_indices[start_idx:end_idx]
        batch_protein_idx = train_protein_indices[start_idx:end_idx]

        batch_max_abs_vec = get_columnwise_max_abs_batch(
            batch_ligand_idx, batch_protein_idx, unique_ligands, unique_proteins
        )
        max_abs_vector = jnp.maximum(max_abs_vector, batch_max_abs_vec)

    scaler_params = {'max_abs': max_abs_vector}
    scaler_params['max_abs'].block_until_ready()
    print("✅ MaxAbs scaler fitted on-device.")

    # --- Phase 2: Heuristic for `t0` learning rate schedule ---
    print("\n--- Estimating t0 using scikit-learn's heuristic ---")
    typw = jnp.sqrt(1.0 / jnp.sqrt(args.alpha))
    dloss_probe = jax.nn.sigmoid(typw)
    eta0 = typw / jnp.maximum(1.0, dloss_probe)
    t0 = 1.0 / (eta0 * args.alpha)
    print(f"✅ Heuristic found eta0={eta0:.4f}, calculated t0={t0:.2f}")

    # --- Phase 3: Initialize Model State ---
    params = {'w': jnp.zeros((1, n_features)), 'b': jnp.zeros(1)}
    opt_state = optimizer.init(params)
    state = (params, opt_state)

    # --- Phase 4: The JAX Training Loop ---
    print(f"\n--- Training Model on JAX ---")

    # Define the body of the training loop for one sample
    def epoch_loop_body(state, idx, lr, scaler_params, train_ligand_indices, train_protein_indices, train_labels, unique_ligands, unique_proteins, alpha, l1_ratio):
        data = ((train_ligand_indices[idx], train_protein_indices[idx]), train_labels[idx])
        new_state, loss = train_step(state, data, lr, scaler_params, unique_ligands, unique_proteins, alpha, l1_ratio)
        return new_state, loss

    # Create a partial function for the loop body with fixed arguments
    loop_body_partial = partial(epoch_loop_body,
                                lr=args.learning_rate,
                                scaler_params=scaler_params,
                                train_ligand_indices=train_ligand_indices,
                                train_protein_indices=train_protein_indices,
                                train_labels=train_labels,
                                unique_ligands=unique_ligands,
                                unique_proteins=unique_proteins,
                                alpha=args.alpha, l1_ratio=args.l1_ratio)

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train_samples)

        # Use jax.lax.scan for a fast, compiled training loop over one epoch
        state, losses = jax.lax.scan(loop_body_partial, state, perm)
        
        losses[-1].block_until_ready() # Block on final loss to get accurate epoch timing
        t0 += n_train_samples # Update t0 for the learning rate schedule

        duration = time.time() - epoch_start_time
        it_per_sec = n_train_samples / duration
        print(f"Epoch {epoch + 1}/{args.epochs} complete in {duration:.2f}s. Effective speed: {it_per_sec:,.0f} it/s")

    # --- Phase 5: Evaluation ---
    if args.test_data:
        print("\n--- Evaluating Model ---")
        test_shard_dir = os.path.join(args.shard_dir, 'test')
        test_ligand_indices = jax.device_put(np.load(os.path.join(test_shard_dir, 'df_ligand_indices.npy')))
        test_protein_indices = jax.device_put(np.load(os.path.join(test_shard_dir, 'df_protein_indices.npy')))
        y_test = np.load(os.path.join(test_shard_dir, 'df_labels.npy'))

        final_params, _ = state 
        
        predictions = predict_probas_batched(
            final_params, 
            scaler_params, # Pass the fitted scaler params
            test_ligand_indices, test_protein_indices,
            unique_ligands, unique_proteins, 
            batch_size=args.predict_batch_size
        )
        
        predictions.block_until_ready() # Block until all GPU prediction work is complete

        score = roc_auc_score(y_test, np.asarray(predictions))
        print("\n--- Results ---")
        print(f"Test Set ROC AUC Score: {score:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ultimate Performance Out-of-Core Classifier with JAX.")
    # Path args
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--protein_features_path', type=str, required=True, help="Path to the source .npz file.")
    parser.add_argument('--shard_dir', type=str, default=None, help="Directory to cache data shards. Defaults to a subdir next to train_data.")
    # Featurization args
    parser.add_argument('--ligand_featurizer', type=str, default='ibm/MoLFormer-XL-both-10pct')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for HuggingFace featurization.")
    parser.add_argument('--fit_batch_size', type=int, default=32, help="Batch size for fitting the JAX StandardScaler.")

    parser.add_argument('--predict_batch_size', type=int, default=32, help="Batch size for inference.")
    # Model args
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--alpha', type=float, default=1e-4, help="Regularization strength (like in sklearn).")
    parser.add_argument('--l1_ratio', type=float, default=0.15, help="Elastic Net mixing parameter (0=L2, 1=L1).")
    # Training loop args
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
