import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import time
import scipy.sparse as sp

# Scikit-learn imports
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# XGBoost for the in-memory path
from xgboost import XGBClassifier

# PyTorch and Transformers for featurization
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

class HuggingFaceFeaturizer:
    """
    Handles featurization of SMILES strings using PyTorch-based Hugging Face models.
    """
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
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings)

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np

# Abbreviation for PartitionSpec
P = PartitionSpec

def generate_sharded_batches_jax(df, ligand_map, protein_map, sharding, batch_size):
    """
    A generator that mimics the logic of `resistance_classifier.py` but yields
    feature-sharded JAX arrays instead of CPU NumPy arrays.

    Args:
        df_iterator: An iterator over a pandas DataFrame (e.g., df.iterrows()).
        ligand_map (dict): Maps SMILES string to its NumPy feature vector.
        protein_map (dict): Maps target_id to its NumPy feature vector.
        sharding (NamedSharding): The JAX sharding rule to apply to the features.
        batch_size (int): The number of samples per batch.
    """
    X_batch_list = []
    y_batch_list = []

    for i, row in enumerate(df.itertuples()):
        # --- 1. Assemble features in NumPy on the CPU ---
        # This part of the logic remains the same as your original code.
        ligand_vec = ligand_map.get(row.Ligand_SMILES)
        protein_vec = protein_map.get(row.target_id)

        if ligand_vec is None or protein_vec is None:
            continue # Skip if features are missing

        # Combine ligand and protein features
        combined_features = np.concatenate([ligand_vec, protein_vec])
        label = row.confers_resistance

        X_batch_list.append(combined_features)
        y_batch_list.append(label)

        # --- 2. When batch is full, convert, shard, and yield ---
        if len(X_batch_list) == batch_size:
            # Convert lists to NumPy arrays
            X_batch_np = np.array(X_batch_list, dtype=np.float32)
            y_batch_np = np.array(y_batch_list, dtype=np.float32)

            # --- THE CRITICAL STEP ---
            # Move data to devices and apply the feature sharding.
            # JAX handles splitting the array along the feature axis.
            X_batch_sharded_jax = jax.device_put(X_batch_np, sharding)

            # Labels are replicated, not sharded.
            y_batch_jax = jax.device_put(y_batch_np)

            yield X_batch_sharded_jax, y_batch_jax

            # Reset for the next batch
            X_batch_list, y_batch_list = [], []

    # Potentially yield the last, smaller batch if it exists
    if X_batch_list:
        X_batch_np = np.array(X_batch_list, dtype=np.float32)
        y_batch_np = np.array(y_batch_list, dtype=np.float32)
        X_batch_sharded_jax = jax.device_put(X_batch_np, sharding)
        y_batch_jax = jax.device_put(y_batch_np)

def main(args):
    start_time = time.time()
    print("--- Training Drug Resistance Classifier ---")

    # --- Load Data & Pre-computed Features ---
    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)

    # Clean column names to be valid identifiers (e.g., 'Ligand SMILES' -> 'Ligand_SMILES')
    train_df.columns = train_df.columns.str.replace(' ', '_')
    test_df.columns = test_df.columns.str.replace(' ', '_')

    # Create target_id for mapping
    train_df['target_id'] = train_df['uniprot_id'] + '_' + train_df['mutation']
    test_df['target_id'] = test_df['uniprot_id'] + '_' + test_df['mutation']

    # --- Shared Feature Preparation ---
    print("Loading protein features...")
    protein_archive = np.load(args.protein_features_path)
    protein_embeddings = {k: protein_archive[k] for k in protein_archive}

    available_targets = set(protein_embeddings.keys())
    train_df = train_df[train_df['target_id'].isin(available_targets)].reset_index(drop=True)
    test_df = test_df[test_df['target_id'].isin(available_targets)].reset_index(drop=True)

    print("Featurizing ligand SMILES...")
    hf_featurizer = HuggingFaceFeaturizer()
    unique_smiles = pd.concat([train_df['Ligand_SMILES'], test_df['Ligand_SMILES']]).unique()
    smiles_embeddings = hf_featurizer.featurize(list(unique_smiles), args.ligand_featurizer, batch_size=args.batch_size)
    smiles_map = {smile: emb for smile, emb in zip(unique_smiles, smiles_embeddings)}

    num_devices = len(jax.devices())
    if num_devices == 0:
        raise RuntimeError("No JAX devices found.")
    print(f"Found {num_devices} devices.")

    device_mesh = Mesh(devices=jax.devices(), axis_names=('features',))
    # Shard the feature dimension (axis 1) across the 'features' mesh axis.
    # Replicate the batch dimension (axis 0).
    feature_sharding = NamedSharding(device_mesh, P(None, 'features'))

    # for df.iterrows()
    train_df = train_df.rename({'Ligand SMILES': 'Ligand_SMILES'})
    test_df = test_df.rename({'Ligand SMILES': 'Ligand_SMILES'})

    print("Generating JAX arrays for train data...")
    # --- 3. Use the Sharded Batch Generator ---
    batch_size = 128
    data_generator = generate_sharded_batches_jax(
        train_df,
        smiles_map,
        protein_embeddings,
        feature_sharding,
        batch_size=batch_size
    )

    data = list(data_generator)
    X_train = jnp.concatenate([batch[0] for batch in data])
    y_train = jnp.concatenate([batch[1] for batch in data])

    from bcd_svm import JaxBCD_SVM
    svm = JaxBCD_SVM(C=1, max_iter=500, random_seed=42, tol=1e-5)

    svm.fit(X_train, y_train)

    print("Generating JAX arrays for test data...")
    # --- 3. Use the Sharded Batch Generator ---
    batch_size = 128
    data_generator = generate_sharded_batches_jax(
        test_df,
        smiles_map,
        protein_embeddings,
        feature_sharding,
        batch_size=batch_size
    )

    data = list(data_generator)
    X_test = jnp.concatenate([batch[0] for batch in data])
    y_test = jnp.concatenate([batch[1] for batch in data])

    predictions = svm.decision_function(X_test)
    print(predictions)

    # --- Final Score ---
    score = roc_auc_score(y_test, predictions)

    print("\n--- Results ---")
    print(f"Test Set ROC AUC Score: {score:.4f}")
    total_time = time.time() - start_time
    print(f"Total pipeline time: {total_time:.2f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a drug resistance classifier.")
    parser.add_argument('--train_data', type=str, required=True, help="Path to training data Parquet file.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to test data Parquet file.")
    parser.add_argument('--protein_features_path', type=str, required=True, help="Path to protein features NPZ file.")
    parser.add_argument('--ligand_featurizer', type=str, default='ibm/MoLFormer-XL-both-10pct', help="Hugging Face model for ligand featurization.")
    parser.add_argument('--model_type', type=str, choices=['xgb', 'liblinear'], default='liblinear', help="Model to train. 'sgd' for out-of-core, 'xgb' for in-memory.")
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for SGD training.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for featurization and SGD training.")
    
    args = parser.parse_args()
    main(args)
