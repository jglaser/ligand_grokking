import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
import time
import json
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
from sklearn.metrics import roc_auc_score

# =============================================================================
# 1. DATA PREPARATION AND DATASET
# =============================================================================
def prepare_shards(df, smiles_map, protein_archive_path, shard_dir):
    print(f"--- Preparing ultimate performance cache in {shard_dir} ---")
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)
    
    protein_archive = np.load(protein_archive_path)
    
    print("Creating global unique feature stores...")
    unique_smiles = df['Ligand SMILES'].unique()
    unique_proteins = df['target_id'].unique()
    
    smiles_to_global_idx = {smile: i for i, smile in enumerate(unique_smiles)}
    protein_to_global_idx = {pid: i for i, pid in enumerate(unique_proteins)}

    np.save(os.path.join(shard_dir, 'unique_ligands.npy'), np.vstack([smiles_map[s] for s in unique_smiles]))
    np.save(os.path.join(shard_dir, 'unique_proteins.npy'), np.vstack([protein_archive[pid] for pid in unique_proteins]))
    with open(os.path.join(shard_dir, 'ligand_index.json'), 'w') as f:
        json.dump(smiles_to_global_idx, f)
    with open(os.path.join(shard_dir, 'protein_index.json'), 'w') as f:
        json.dump(protein_to_global_idx, f)

    print("Converting DataFrame columns to fast NumPy arrays...")
    np.save(os.path.join(shard_dir, 'df_ligand_smiles.npy'), df['Ligand SMILES'].values)
    np.save(os.path.join(shard_dir, 'df_target_ids.npy'), df['target_id'].values)
    np.save(os.path.join(shard_dir, 'df_labels.npy'), df['is_effective_against_mutant'].values.astype(np.int64))
        
    print("✅ Cache preparation complete.")

class ShardedDataset(Dataset):
    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        self.stores_loaded = False
        self.df_ligand_smiles, self.df_target_ids, self.df_labels = None, None, None
        self.global_ligand_vectors, self.global_protein_vectors = None, None
        self.smiles_to_global_idx, self.protein_to_global_idx = None, None
        self._len = len(np.load(os.path.join(self.shard_dir, 'df_labels.npy'), mmap_mode='r'))

    def __len__(self):
        return self._len

    def _load_stores(self):
        self.global_ligand_vectors = np.load(os.path.join(self.shard_dir, 'unique_ligands.npy'))
        self.global_protein_vectors = np.load(os.path.join(self.shard_dir, 'unique_proteins.npy'))
        with open(os.path.join(self.shard_dir, 'ligand_index.json'), 'r') as f:
            self.smiles_to_global_idx = json.load(f)
        with open(os.path.join(self.shard_dir, 'protein_index.json'), 'r') as f:
            self.protein_to_global_idx = json.load(f)
        
        self.df_ligand_smiles = np.load(os.path.join(self.shard_dir, 'df_ligand_smiles.npy'), allow_pickle=True)
        self.df_target_ids = np.load(os.path.join(self.shard_dir, 'df_target_ids.npy'), allow_pickle=True)
        self.df_labels = np.load(os.path.join(self.shard_dir, 'df_labels.npy'), mmap_mode='r')
        self.stores_loaded = True

    def __getitem__(self, idx):
        if not self.stores_loaded:
            self._load_stores()

        smiles = self.df_ligand_smiles[idx]
        target_id = self.df_target_ids[idx]
        label = self.df_labels[idx]
        
        ligand_idx = self.smiles_to_global_idx[smiles]
        protein_idx = self.protein_to_global_idx[target_id]
        
        ligand_vec = self.global_ligand_vectors[ligand_idx]
        protein_vec = self.global_protein_vectors[protein_idx]
        
        features = np.concatenate([ligand_vec, protein_vec]).astype(np.float32)
        
        return features, np.int64(label)

    @staticmethod
    def worker_init_fn(worker_id):
        """
        FIX: Robust initialization function for each DataLoader worker.
        """
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset
        dataset._load_stores()

# =============================================================================
# 2. GPU-NATIVE SCALER AND CLASSIFIER
# =============================================================================
class GPUStandardScaler:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_ = None
        self.scale_ = None

    def fit(self, df, shard_dir):
        print(f"Fitting Redundancy-Aware GPUStandardScaler on {self.device}...")
        
        ligand_vectors = np.load(os.path.join(shard_dir, 'unique_ligands.npy'))
        protein_vectors = np.load(os.path.join(shard_dir, 'unique_proteins.npy'))
        with open(os.path.join(shard_dir, 'ligand_index.json'), 'r') as f:
            smiles_to_global_idx = json.load(f)
        with open(os.path.join(shard_dir, 'protein_index.json'), 'r') as f:
            protein_to_global_idx = json.load(f)
        
        ligand_counts = df['Ligand SMILES'].value_counts()
        protein_counts = df['target_id'].value_counts()
        
        n_samples = len(df)
        n_features = ligand_vectors.shape[1] + protein_vectors.shape[1]

        total_sum = torch.zeros(n_features, dtype=torch.float64, device=self.device)
        total_sum_sq = torch.zeros_like(total_sum)

        ligand_indices = [smiles_to_global_idx[s] for s in ligand_counts.index]
        ligand_features_gpu = torch.from_numpy(ligand_vectors[ligand_indices]).to(self.device, dtype=torch.float64)
        counts_gpu = torch.tensor(ligand_counts.values, device=self.device, dtype=torch.float64).unsqueeze(1)
        total_sum[:ligand_vectors.shape[1]] = torch.sum(ligand_features_gpu * counts_gpu, dim=0)
        total_sum_sq[:ligand_vectors.shape[1]] = torch.sum((ligand_features_gpu**2) * counts_gpu, dim=0)
        
        protein_indices = [protein_to_global_idx[pid] for pid in protein_counts.index]
        protein_features_gpu = torch.from_numpy(protein_vectors[protein_indices]).to(self.device, dtype=torch.float64)
        counts_gpu = torch.tensor(protein_counts.values, device=self.device, dtype=torch.float64).unsqueeze(1)
        total_sum[ligand_vectors.shape[1]:] = torch.sum(protein_features_gpu * counts_gpu, dim=0)
        total_sum_sq[ligand_vectors.shape[1]:] = torch.sum((protein_features_gpu**2) * counts_gpu, dim=0)
        
        self.mean_ = (total_sum / n_samples).to(torch.float32)
        variance = (total_sum_sq / n_samples) - (self.mean_.to(torch.float64)**2)
        self.scale_ = torch.sqrt(variance.clamp(min=1e-7)).to(torch.float32)
        print("Scaler fitting complete.")
        return self

class _LogisticModule(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

class PyTorchLogisticRegression:
    def __init__(self, learning_rate=0.001, alpha=0.0001, l1_ratio=0.15,
                 random_seed=42, dtype='float32', quant_scale=None, device=None,
                 predict_batch_size=4096):
        self.learning_rate, self.alpha, self.l1_ratio = learning_rate, alpha, l1_ratio
        self.random_seed, self.dtype, self.quant_scale = random_seed, dtype, quant_scale
        self.device, self.predict_batch_size = device, predict_batch_size
        
        dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int8': torch.int8}
        if self.dtype not in dtype_map: raise ValueError("Unsupported dtype")
        self.dtype_ = dtype_map[self.dtype]

        if self.dtype == 'int8' and self.quant_scale is None:
            self.quant_scale = 4.0 / 127.0
            print(f"Warning: `quant_scale` not provided. Using default: {self.quant_scale:.4f}")

    def _initialize(self, X, y):
        torch.manual_seed(self.random_seed)
        self.device_ = torch.device(self.device)
        self.classes_, self.n_features_in_ = torch.unique(y), X.shape[1]
        model_dtype = torch.float32 if self.dtype == 'int8' else self.dtype_
        
        # Instantiate the base model
        model = _LogisticModule(self.n_features_in_).to(device=self.device_, dtype=model_dtype)
        # --- THE FINAL OPTIMIZATION: COMPILE THE MODEL ---
        print("Compiling model with torch.compile()...")
        self.model_ = torch.compile(model)
        
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        self.criterion_ = nn.BCEWithLogitsLoss()
        if self.dtype == 'int8':
            self.model_.register_buffer('quant_scale_', torch.tensor(self.quant_scale, device=self.device_))
        self.is_fitted_ = True

    def partial_fit(self, X_batch, y_batch):
        if not hasattr(self, 'model_'): self._initialize(X_batch, y_batch)
        if self.dtype == 'int8':
            X_batch_final = X_batch.to(self.model_.linear.weight.dtype) * self.model_.quant_scale_
        else:
            X_batch_final = X_batch

        self.model_.train()
        y_tensor = y_batch.to(self.model_.linear.weight.dtype).view(-1, 1)
        self.optimizer_.zero_grad()
        outputs = self.model_(X_batch_final)
        loss = self.criterion_(outputs, y_tensor)
        
        if self.alpha > 0:
            l1_penalty = sum(torch.linalg.vector_norm(p, ord=1) for p in self.model_.parameters() if p.dim() > 1)
            l2_penalty = sum(torch.linalg.vector_norm(p, ord=2).pow(2) for p in self.model_.parameters() if p.dim() > 1)
            # --- FIX: Corrected typo from l1_ratio * l1_ratio to self.l1_ratio ---
            loss += self.alpha * (self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * 0.5 * l2_penalty)
        
        loss.backward()
        self.optimizer_.step()
        return self

# =============================================================================
# 3. HUGGING FACE FEATURIZER
# =============================================================================
class HuggingFaceFeaturizer:
    # (Unchanged)
    def __init__(self):
        self.models_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"HuggingFaceFeaturizer using device: {self.device}")

    def _load_model(self, model_name):
        if model_name not in self.models_cache:
            print(f"Loading and caching model: {model_name}...")
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
# 4. MAIN SCRIPT LOGIC
# =============================================================================
def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Loading and filtering dataframes ---")
    train_df_full = pd.read_parquet(args.train_data)
    train_df_full['target_id'] = train_df_full['uniprot_id'] + '_' + train_df_full['mutation']
    with np.load(args.protein_features_path) as archive:
        available_targets = set(archive.files)
    train_df = train_df_full[train_df_full['target_id'].isin(available_targets)].reset_index(drop=True)
    
    if args.shard_dir is None:
        base_dir = os.path.dirname(args.train_data)
        args.shard_dir = os.path.join(base_dir, "shard_cache")
    
    canary_file = os.path.join(args.shard_dir, 'unique_ligands.npy')
    
    regen_needed = not os.path.exists(canary_file)
    if not regen_needed:
        if os.path.getmtime(args.train_data) > os.path.getmtime(canary_file) or \
           os.path.getmtime(args.protein_features_path) > os.path.getmtime(canary_file):
            regen_needed = True

    if regen_needed:
        hf_featurizer = HuggingFaceFeaturizer()
        unique_smiles = train_df['Ligand SMILES'].unique()
        smiles_map = {s: emb for s, emb in zip(unique_smiles, hf_featurizer.featurize(unique_smiles, args.ligand_featurizer, batch_size=args.batch_size))}
        prepare_shards(train_df, smiles_map, args.protein_features_path, args.shard_dir)
    else:
        print(f"✅ Using existing performance cache at {args.shard_dir}")

    print("\n--- Phase 1: Fitting Scaler ---")
    scaler = GPUStandardScaler(device=device)
    scaler.fit(train_df, args.shard_dir)
    print(f"✅ Scaler fit. Total features: {scaler.mean_.shape[0]:,}")
    
    print(f"\n--- Phase 2: Training Model on {device} ---")
    model = PyTorchLogisticRegression(
        device=device, dtype=args.dtype, learning_rate=args.learning_rate,
        alpha=args.alpha, l1_ratio=args.l1_ratio
    )
    
    train_dataset = ShardedDataset(args.shard_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=ShardedDataset.worker_init_fn
    )
    
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            X_batch_scaled = (X_batch - scaler.mean_) / scaler.scale_
            
            if model.dtype == 'int8':
                X_batch_quantized = torch.clamp(torch.round(X_batch_scaled / model.quant_scale), -128, 127).to(torch.int8)
                X_batch_final = X_batch_quantized
            else:
                X_batch_final = X_batch_scaled.to(model.dtype_)
            
            model.partial_fit(X_batch_final, y_batch)

    print("\n--- Phase 3: Evaluating Model ---")
    
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
    
    # Model args
    parser.add_argument('--dtype', type=str, choices=['float32', 'float16', 'bfloat16', 'int8'], default='float32')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=1e-4, help="Regularization strength")
    parser.add_argument('--l1_ratio', type=float, default=0.15, help="Elastic Net mixing parameter")
    
    # Training loop args
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8, help="Number of worker processes for DataLoader")
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
