import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import torch

class HuggingFaceFeaturizer:
    """
    A class to handle loading, caching, and using Hugging Face models for featurization.
    This prevents loading the same model into memory multiple times.
    Can be used for both protein sequences and SMILES strings.
    """
    def __init__(self):
        print("Initializing HuggingFaceFeaturizer...")
        self.models_cache = {} # Cache for storing loaded models and tokenizers
        self.is_rostlab = None
        try:
            import torch_xla
            self.device = torch_xla.device()
            self.compile_backend = 'openxla'
            print("TPU found. Using OpenXLA backend.")
        except ImportError:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.compile_backend = 'inductor'
            print(f"Using device: {self.device} with Inductor backend.")

    def _load_model(self, model_name):
        """
        Loads a model and tokenizer if not already in the cache.
        """
        if model_name not in self.models_cache:
            print(f"Loading and caching model: {model_name}...")
            self.is_rostlab = 'Rostlab' in model_name
            if 't5' in model_name:
                model = T5EncoderModel.from_pretrained(model_name).to(self.device).eval()
                tokenizer = T5Tokenizer.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

#            try:
#                model = torch.compile(model, backend=self.compile_backend)
#                print("Model compiled successfully.")
#            except Exception as e:
#                print(f"Warning: Model compilation failed. Proceeding without compilation. Error: {e}")

            self.models_cache[model_name] = {'model': model, 'tokenizer': tokenizer}
        return self.models_cache[model_name]

    def featurize(self, sequences, model_name, batch_size=32):
        """
        Generates embeddings for a list of sequences (or SMILES) using a specified Hugging Face model.
        """
        model_dict = self._load_model(model_name)
        model = model_dict['model']
        tokenizer = model_dict['tokenizer']

        all_embeddings = []
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Featurizing with {os.path.basename(model_name)}"):
            batch_seqs = sequences[i:i+batch_size]

            if self.is_rostlab:
                batch_seqs = [" ".join(list(seq)) for seq in batch_seqs]

            inputs = tokenizer(batch_seqs, return_tensors='pt', padding='max_length', max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

def generate_feature_batches(df, ligand_map, protein_map, batch_size):
    """
    A generator that yields batches of combined features and labels.
    This is crucial for out-of-core learning with partial_fit.
    """
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        X_ligand_batch = np.array([ligand_map[s] for s in batch_df['Ligand SMILES']])
        X_protein_batch = np.array([protein_map[tid] for tid in batch_df['target_id']])
        
        X_batch = np.concatenate([X_ligand_batch, X_protein_batch], axis=1)
        y_batch = batch_df['is_effective_against_mutant'].values
        
        yield X_batch, y_batch

def main(args):
    print("--- Training Drug Resistance Classifier ---")
    # --- Load Data & Pre-computed Features ---
    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)

    print(f"Loading pre-computed delta vectors from {args.protein_features_path}...")
    delta_vectors_archive = np.load(args.protein_features_path)
    
    train_df['target_id'] = train_df['uniprot_id'] + '_' + train_df['mutation']
    test_df['target_id'] = test_df['uniprot_id'] + '_' + test_df['mutation']
    
    available_targets = set(delta_vectors_archive.files)
    train_df = train_df[train_df['target_id'].isin(available_targets)]
    test_df = test_df[test_df['target_id'].isin(available_targets)]

    # --- Protein Feature Handling ---
    unique_target_ids = pd.concat([train_df['target_id'], test_df['target_id']]).unique()
    unique_vectors = np.array([delta_vectors_archive[tid] for tid in unique_target_ids])
    
    if args.projection_components and args.model_type != 'sgd':
        print(f"Applying Sparse Random Projection to reduce features to {args.projection_components} components...")
        projector = SparseRandomProjection(n_components=args.projection_components, random_state=args.random_seed)
        projected_unique_vectors = projector.fit_transform(unique_vectors)
    else:
        projected_unique_vectors = unique_vectors

    protein_feature_map = {tid: vec for tid, vec in zip(unique_target_ids, projected_unique_vectors)}

    # --- Ligand Feature Handling ---
    hf_featurizer = HuggingFaceFeaturizer()
    unique_smiles = pd.concat([train_df['Ligand SMILES'], test_df['Ligand SMILES']]).unique()
    smiles_map = {s: emb for s, emb in zip(unique_smiles, hf_featurizer.featurize(list(unique_smiles), args.ligand_featurizer))}

    # --- Model Training ---
    if args.model_type == 'sgd':
        print("\nTraining SGDClassifier using memory-efficient partial_fit...")
        scaler = StandardScaler()
        model = SGDClassifier(loss='log_loss', penalty='elasticnet', random_state=args.random_seed, max_iter=1000, tol=1e-3)
        
        # Get all possible classes for the first partial_fit call
        all_classes = np.array([0, 1])

        for epoch in range(args.epochs):
            print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
            batch_generator = generate_feature_batches(train_df, smiles_map, protein_feature_map, args.batch_size)
            for X_batch, y_batch in tqdm(batch_generator, total=int(np.ceil(len(train_df)/args.batch_size))):
                scaler.partial_fit(X_batch)
                X_scaled_batch = scaler.transform(X_batch)
                model.partial_fit(X_scaled_batch, y_batch, classes=all_classes)
        
        # Evaluation using the trained model
        print("Evaluating model on test set in batches...")
        all_predictions = []
        all_true_labels = []
        test_generator = generate_feature_batches(test_df, smiles_map, protein_feature_map, args.batch_size)
        for X_batch, y_batch in test_generator:
            X_scaled_batch = scaler.transform(X_batch)
            all_predictions.append(model.decision_function(X_scaled_batch))
            all_true_labels.append(y_batch)
        
        predictions = np.concatenate(all_predictions)
        y_test = np.concatenate(all_true_labels)

    elif args.model_type == 'xgb':
        print("\nLoading all features into memory for XGBoost training...")
        X_protein_train = np.array([protein_feature_map[tid] for tid in train_df['target_id']])
        X_protein_test = np.array([protein_feature_map[tid] for tid in test_df['target_id']])
        X_ligand_train = np.array([smiles_map[s] for s in train_df['Ligand SMILES']])
        X_ligand_test = np.array([smiles_map[s] for s in test_df['Ligand SMILES']])
        
        X_train = np.concatenate([X_ligand_train, X_protein_train], axis=1)
        X_test = np.concatenate([X_ligand_test, X_protein_test], axis=1)
        y_train = train_df['is_effective_against_mutant'].values
        y_test = test_df['is_effective_against_mutant'].values

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
        print("Training XGBoost model...")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict_proba(X_test)[:, 1]

    score = roc_auc_score(y_test, predictions)
    
    print("\n--- Results ---")
    print(f"Test Set ROC AUC Score: {score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a drug resistance classifier.")
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--protein_features_path', type=str, required=True)
    parser.add_argument('--ligand_featurizer', type=str, default='ibm/MoLFormer-XL-both-10pct')
    parser.add_argument('--projection_components', type=int, default=None)
    parser.add_argument('--model_type', type=str, choices=['xgb', 'sgd'], default='xgb')
    parser.add_argument('--random_seed', type=int, default=42)
    # New arguments for SGD partial_fit
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for SGD training.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for SGD training.")
    args = parser.parse_args()
    main(args)


