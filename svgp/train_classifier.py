import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
from sklearn.metrics import roc_auc_score

# --- Logger Abstraction ---
class CSVLogger:
    def __init__(self, config, log_file):
        self.config = config
        self.log_file = log_file
        self.history = []
        self.header_written = False # Flag to ensure header is written only once
        
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Start with a clean file and write the config as a comment
        with open(self.log_file, 'w') as f:
            f.write(f"# Config: {self.config}\n")

    def log(self, metrics, step):
        log_entry = {'epoch': step, **metrics}
        self.history.append(log_entry)
        if len(self.history) % 100 == 0:
            self.flush()

    def flush(self):
        if self.history:
            df = pd.DataFrame(self.history)
            # This logic ensures the header is written exactly once.
            df.to_csv(self.log_file, mode='a', index=False, header=not self.header_written)
            self.header_written = True # Set flag after the first write
            self.history = []

    def finish(self):
        self.flush() # Write any remaining history
        print(f"Training log saved to: {self.log_file}")

# --- Featurization ---
from transformers import logging as hf_logging
import torch

hf_logging.set_verbosity_error()

def featurize_smiles(smiles_list, model_name, batch_size=64, max_length=512):
    print(f"Loading ligand featurizer: {model_name}...")
    try:
        import torch_xla
        device = torch_xla.device()
        compile_backend = 'openxla'
        print("TPU found.")
    except:
        device = 0 if torch.cuda.is_available() else -1
        compile_backend = 'inductor'

    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model = torch.compile(model, backend=compile_backend)

    print("Featurizing SMILES...")
    all_features = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Featurizing Ligands"):
        batch_smiles = [s for s in smiles_list[i:i + batch_size]]
        if len(batch_smiles) != batch_size:
            batch_smiles += [''] * (batch_size - len(batch_smiles))

        inputs = tokenizer(batch_smiles, max_length=max_length, padding='max_length', truncation=True,
                           return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs.to(device)).pooler_output
        all_features.append(outputs.cpu().numpy())
    return np.concatenate(all_features)[:len(smiles_list)]

def featurize_proteins(sequence_list, model_name, batch_size=16, max_length=2048):
    """
    Generates embeddings for a list of protein sequences using a protein language model.
    """
    print(f"Loading protein featurizer: {model_name}...")
    try:
        import torch_xla
        device = torch_xla.device()
        compile_backend = 'openxla'
        print("TPU found.")
    except:
        device = 0 if torch.cuda.is_available() else -1
        compile_backend = 'inductor'

    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model = torch.compile(model, backend=compile_backend)

    print("Featurizing Protein Sequences...")
    sequences_with_spaces = [" ".join(list(seq)) for seq in sequence_list]

    all_features = []
    for i in tqdm(range(0, len(sequences_with_spaces), batch_size), desc="Featurizing Proteins"):
        batch_seqs = sequences_with_spaces[i:i + batch_size]
        if len(batch_seqs) != batch_size:
            batch_seqs += [''] * (batch_size - len(batch_seqs))

        inputs = tokenizer(batch_seqs, max_length=max_length, padding='max_length', truncation=True,
                           return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs.to(device)).last_hidden_state
        batch_features = np.mean(outputs.cpu().numpy(), axis=1)
        all_features.append(batch_features)
    return np.concatenate(all_features)[:len(sequences_with_spaces)]


# --- SVGP and Training Logic ---
from svgp_classifier import SparseVariationalGPClassifier

class LinearEncoder:
    def __init__(self, key, input_dim, output_dim):
        import jax
        import jax.numpy as jnp
        self.params = {'W': jax.random.normal(key, shape=(input_dim, output_dim)) * jnp.sqrt(2.0 / input_dim)}
        self.output_dim = output_dim
    def __call__(self, params, x):
        return x @ params['W']

def log_callback_factory(X_train, y_train, X_val, y_val, logger):
    def log_callback(model, epoch, metrics, params, bias_state):
        if epoch % 100 == 0:
            # --- THE FIX: Use the dedicated score_auc method from the classifier ---
            
            # --- Validation Metrics ---
            metrics["validation_accuracy"] = model.score(X_val, y_val, params=params)
            metrics["validation_auc"] = model.score_auc(X_val, y_val, params=params)

            # --- Training Metrics (on a subset) ---
            train_subset_idx = np.random.choice(X_train.shape[0], size=min(len(y_val), len(y_train)), replace=False)
            X_train_subset, y_train_subset = X_train[train_subset_idx], y_train[train_subset_idx]
            
            metrics["train_accuracy"] = model.score(X_train_subset, y_train_subset, params=params)
            metrics["train_auc"] = model.score_auc(X_train_subset, y_train_subset, params=params)

            logger.log(metrics, step=epoch)
        return metrics
    return log_callback

def main():
    parser = argparse.ArgumentParser(description="Train a multi-target proteochemometric SVGP classifier.")
    parser.add_argument("train_file", type=str, help="Path to the training CSV file.")
    parser.add_argument("test_file", type=str, help="Path to the testing CSV file.")
    
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--n_inducing_points", type=int, default=100)
    parser.add_argument("--encoder_dim", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1e-3)
    parser.add_argument("--jitter", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, required=True, help="Full path for the output CSV log file.")
    parser.add_argument("--ligand_featurizer", type=str, default="ibm/MoLFormer-XL-both-10pct")
    parser.add_argument("--protein_featurizer", type=str, default="Rostlab/prot_bert_bfd")
    parser.add_argument("--featurizer_mode", type=str, default="concat", choices=["ligand", "protein", "concat"])

    args = parser.parse_args()

    logger = CSVLogger(config=vars(args), log_file=args.log_file)

    try:
        print("Loading and filtering data with pandas...")
        train_df = pd.read_csv(args.train_file).dropna(subset=['smiles', 'sequence'])
        test_df = pd.read_csv(args.test_file).dropna(subset=['smiles', 'sequence'])

        # --- Featurize Data based on Mode ---
        X_ligand_train, X_ligand_test = None, None
        if args.featurizer_mode in ['ligand', 'concat']:
            X_ligand_train = featurize_smiles(train_df['smiles'].tolist(), model_name=args.ligand_featurizer)
            X_ligand_test = featurize_smiles(test_df['smiles'].tolist(), model_name=args.ligand_featurizer)
        
        X_protein_train, X_protein_test = None, None
        if args.featurizer_mode in ['protein', 'concat']:
            all_sequences = pd.concat([train_df['sequence'], test_df['sequence']])
            unique_sequences = all_sequences.unique().tolist()
            
            print(f"\nFound {len(unique_sequences)} unique protein sequences to featurize.")
            unique_protein_embeddings = featurize_proteins(unique_sequences, model_name=args.protein_featurizer)
            
            seq_to_embedding_map = {seq: emb for seq, emb in zip(unique_sequences, unique_protein_embeddings)}
            
            print("Mapping protein embeddings to ligands...")
            X_protein_train = np.array([seq_to_embedding_map[seq] for seq in train_df['sequence']])
            X_protein_test = np.array([seq_to_embedding_map[seq] for seq in test_df['sequence']])

        import jax
        key = jax.random.PRNGKey(args.random_seed)

        # --- Combine features based on the selected mode ---
        if args.featurizer_mode == 'concat':
            X_train = np.concatenate([X_ligand_train, X_protein_train], axis=1)
            X_test = np.concatenate([X_ligand_test, X_protein_test], axis=1)
        elif args.featurizer_mode == 'ligand':
            X_train, X_test = X_ligand_train, X_ligand_test
        elif args.featurizer_mode == 'protein':
            X_train, X_test = X_protein_train, X_protein_test

        y_train = train_df['active'].values
        y_test = test_df['active'].values

        # --- Setup and Train Model ---
        callback = log_callback_factory(X_train, y_train, X_test, y_test, logger)
        key, encoder_key, hparam_key = jax.random.split(key, 3)
        
        input_dim = X_train.shape[-1]
        encoder = LinearEncoder(encoder_key, input_dim, args.encoder_dim)
        
        log_gamma_init = jax.random.normal(hparam_key, shape=(args.encoder_dim,))
        hparams_init = {'log_gamma': log_gamma_init}

        svgpc = SparseVariationalGPClassifier(
            learning_rate=args.learning_rate, n_inducing_points=args.n_inducing_points,
            sampler='sgld', temperature=args.temperature, train_inducing_points=True,
            ard=True, log_callback=callback, jitter=args.jitter, encoder=encoder,
            encoder_params=encoder.params, kernel_hparams_init=hparams_init
        )
        
        print(f"Starting training with feature shape: {X_train.shape}")
        svgpc.fit(X_train, y_train, epochs=args.epochs, batch_size=min(args.batch_size, len(y_train)), random_seed=args.random_seed)
        
        print("Training complete.")

    finally:
        logger.finish()

if __name__ == '__main__':
    main()


