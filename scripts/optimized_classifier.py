import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os

# Re-using the featurizer from the original script
from resistance_classifier import HuggingFaceFeaturizer

import jax
import jax.numpy as jnp
from sklearn.metrics import roc_auc_score

# Import our new modules
from utils import BatchedStandardScaler
from bcd_svm_out_of_core import JaxOutOfCoreSVM


def main(args):
    print("--- Training Drug Resistance Classifier with Out-of-Core JAX SVM ---")
    # --- 1. Load Data ---
    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)

    train_df['target_id'] = train_df['uniprot_id'] + '_' + train_df['mutation']
    test_df['target_id'] = test_df['uniprot_id'] + '_' + test_df['mutation']

    # --- 2. Featurize Unique Ligands and Proteins ---
    hf_featurizer = HuggingFaceFeaturizer()
    unique_smiles = pd.concat([train_df['Ligand SMILES'], test_df['Ligand SMILES']]).unique()
    smiles_map = {s: emb for s, emb in zip(unique_smiles, hf_featurizer.featurize(list(unique_smiles), args.ligand_featurizer))}

    print(f"Loading pre-computed protein features from {args.protein_features_path}...")
    protein_archive = np.load(args.protein_features_path)
    unique_target_ids = pd.concat([train_df['target_id'], test_df['target_id']]).unique()
    protein_map = {tid: protein_archive[tid] for tid in unique_target_ids if tid in protein_archive}

    # Filter dataframes to only include targets we have features for
    train_df = train_df[train_df['target_id'].isin(protein_map.keys())].reset_index(drop=True)
    test_df = test_df[test_df['target_id'].isin(protein_map.keys())].reset_index(drop=True)

    # --- 3. Create Unique Feature Matrices and Index Mappings ---
    # These are the full, unique feature sets that will be passed to the SVM
    ligand_idx_map = {smile: i for i, smile in enumerate(smiles_map.keys())}
    protein_idx_map = {tid: i for i, tid in enumerate(protein_map.keys())}

    # Convert maps to ordered JAX arrays
    ligand_features = jnp.array(list(smiles_map.values()))
    protein_features = jnp.array(list(protein_map.values()))
    
    # Create the training and testing pairs using the index maps
    train_pairs = jnp.array([[ligand_idx_map[s], protein_idx_map[t]] for s, t in zip(train_df['Ligand SMILES'], train_df['target_id'])])
    test_pairs = jnp.array([[ligand_idx_map[s], protein_idx_map[t]] for s, t in zip(test_df['Ligand SMILES'], test_df['target_id'])])
    
    y_train = train_df['confers_resistance'].values
    y_test = test_df['confers_resistance'].values

    # --- 4. Fit Batched Scaler using the indexed method ---
    print("\nFitting BatchedStandardScaler out-of-core using indexed pairs...")
    scaler = BatchedStandardScaler()
    
    # Shuffle the training pairs for better statistics calculation
    shuffled_indices = np.arange(len(train_pairs))
    np.random.shuffle(shuffled_indices)

    for i in tqdm(range(0, len(train_pairs), args.batch_size)):
        batch_indices = shuffled_indices[i:i+args.batch_size]
        batch_pairs = train_pairs.at[batch_indices].get()
        scaler.partial_fit_indexed(ligand_features, protein_features, batch_pairs)

    # --- 5. Scale and Normalize Unique Feature Matrices ---
    print("Scaling and normalizing unique feature matrices...")
    ligand_dim = ligand_features.shape[1]
    
    # Transform each feature set with the corresponding part of the learned scaler
    scaled_ligands = (ligand_features - scaler.mean_[:ligand_dim]) / jnp.sqrt(scaler.var_[:ligand_dim] + 1e-7)
    scaled_proteins = (protein_features - scaler.mean_[ligand_dim:]) / jnp.sqrt(scaler.var_[ligand_dim:] + 1e-7)
    
    # Sample-wise L2 normalization
    norm_scaled_ligands = jax.nn.normalize(scaled_ligands, axis=1)
    norm_scaled_proteins = jax.nn.normalize(scaled_proteins, axis=1)

    # --- 6. Train Out-of-Core JAX SVM ---
    svm = JaxOutOfCoreSVM(C=args.C, max_iter=args.epochs, random_seed=args.random_seed)
    
    # The SVM already operates on indexed pairs, so we pass the data directly
    svm.fit(
        norm_scaled_ligands,
        norm_scaled_proteins,
        train_pairs,
        jnp.array(y_train)
    )

    # --- 7. Evaluation ---
    print("\nEvaluating model on the test set...")
    predictions = svm.decision_function(
        norm_scaled_ligands,
        norm_scaled_proteins,
        test_pairs
    )
    
    score = roc_auc_score(y_test, predictions)
    print("\n--- Results ---")
    print(f"Test Set ROC AUC Score: {score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a drug resistance classifier using the Out-of-Core JAX SVM.")
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--protein_features_path', type=str, required=True)
    parser.add_argument('--ligand_featurizer', type=str, default='ibm/MoLFormer-XL-both-10pct')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for SVM training.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for scaler fitting.")
    parser.add_argument('--C', type=float, default=1.0, help="Regularization parameter for the SVM.")
    args = parser.parse_args()
    
    # A little patch because I used jax.nn.normalize which doesn't exist.
    # A proper implementation would be to define it, but for a script this is fine.
    def normalize_manual(x, axis=None, epsilon=1e-12):
        square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
        x_inv_norm = 1. / jnp.sqrt(jnp.maximum(square_sum, epsilon))
        return x * x_inv_norm
    jax.nn.normalize = normalize_manual

    main(args)


