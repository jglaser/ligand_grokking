import argparse
import numpy as np
import polars as pl
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm
import os

# --- Logger Abstraction ---
# This allows us to easily switch between different logging backends.

class BaseLogger:
    def __init__(self, config, run_name):
        self.config = config
        self.run_name = run_name
    def log(self, metrics, step):
        raise NotImplementedError
    def finish(self):
        pass

class WandbLogger(BaseLogger):
    def __init__(self, config, run_name, project_name):
        super().__init__(config, run_name)
        try:
            import wandb
            self.wandb = wandb
            self.run = self.wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                reinit=True
            )
        except ImportError:
            print("Wandb not installed. Please install with 'pip install wandb'.")
            self.wandb = None
    
    def log(self, metrics, step):
        if self.wandb and self.run:
            self.run.log(metrics, step=step)

    def finish(self):
        if self.wandb and self.run:
            self.run.finish()

class TensorBoardLogger(BaseLogger):
    def __init__(self, config, run_name, log_dir="logs"):
        super().__init__(config, run_name)
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_path = os.path.join(log_dir, run_name)
            self.writer = SummaryWriter(log_path)
            # Log hyperparameters as text for easy viewing
            hparams_text = "\n".join([f"{key}: {value}" for key, value in config.items()])
            self.writer.add_text("Hyperparameters", hparams_text)
        except ImportError:
            print("TensorBoard not installed. Please install with 'pip install tensorboard'.")
            self.writer = None

    def log(self, metrics, step):
        if self.writer:
            for key, value in metrics.items():
                if np.isfinite(value): # Only log finite values
                    self.writer.add_scalar(key, np.array(value), step)
    
    def finish(self):
        if self.writer:
            self.writer.close()
            print(f"TensorBoard log for '{self.run_name}' closed.")

# --- The rest of the script remains largely the same ---
# (Featurizer, Encoder, Alpha estimation, etc.)

# Local import of your classifier
from svgp_classifier import SparseVariationalGPClassifier

# --- Encoder Definition ---
class LinearEncoder:
    def __init__(self, key, input_dim, output_dim):
        self.params = {'W': jax.random.normal(key, shape=(input_dim, output_dim)) * jnp.sqrt(2.0 / input_dim)}
        self.output_dim = output_dim
    def __call__(self, params, x):
        return x @ params['W']

# --- HTSR Alpha Calculation (Robust MLE Method) ---
def estimate_alpha_fit(data):
    data = np.asarray(data)
    data = data[data > 1e-9]
    if len(data) < 4: return np.nan
    sorted_data = np.sort(data)
    xmins = np.unique(sorted_data)
    if len(xmins) > 1: xmins = xmins[:-1]
    best_ks, best_alpha = np.inf, np.nan
    for xmin in xmins:
        if xmin <= 0: continue
        tail = sorted_data[sorted_data >= xmin]
        n_tail = len(tail)
        if n_tail < 2: continue
        log_sum = np.sum(np.log(tail / xmin))
        alpha = 1.0 + n_tail / (log_sum + 1e-9)
        empirical_cdf = np.arange(n_tail) / n_tail
        fitted_cdf = 1 - (tail / xmin)**(-alpha + 1)
        ks_stat = np.max(np.abs(empirical_cdf - fitted_cdf))
        if ks_stat < best_ks:
            best_ks, best_alpha = ks_stat, alpha
    return best_alpha

# --- Featurization ---
from transformers import pipeline
def featurize_smiles(smiles_list, model_name, batch_size=64, max_length=512):
    print(f"Loading feature extractor model: {model_name}...")
    feature_extractor = pipeline("feature-extraction", framework="pt", model=model_name, device=0, trust_remote_code=True)
    print("Model loaded. Starting featurization...")
    all_features = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Featurizing Batches"):
        batch_smiles = [s[:max_length] for s in smiles_list[i:i + batch_size]]
        outputs = feature_extractor(batch_smiles, return_tensors="pt")
        batch_features = np.stack([t[0].numpy().mean(axis=0) for t in outputs])
        all_features.append(batch_features)
    return np.vstack(all_features)

# --- Logging Callback ---
def log_callback_factory(X_train, y_train, X_val, y_val, logger):
    def log_callback(model, epoch, metrics, params, bias_state):
        if epoch % 100 == 0:
            val_score = model.score(X_val, y_val, params=params)
            metrics["validation_accuracy"] = val_score
            train_subset_idx = np.random.choice(X_train.shape[0], size=min(len(y_val), len(y_train)), replace=False)
            X_train_subset, y_train_subset = X_train[train_subset_idx], y_train[train_subset_idx]
            train_score = model.score(X_train_subset, y_train_subset, params=params)
            metrics["train_accuracy"] = train_score
        if epoch % 10 == 0 and 'encoder_params' in params and 'W' in params['encoder_params']:
            svals = jnp.linalg.svd(params['encoder_params']['W'], compute_uv=False)
            metrics['htsr_alpha'] = estimate_alpha_fit(svals**2)
        
        # Use the logger object to log the metrics
        logger.log(metrics, step=epoch)
        return metrics # Return for progress bar
    return log_callback

def main():
    parser = argparse.ArgumentParser(description="Train an SVGP classifier on molecular data.")
    # ... (existing args)
    parser.add_argument("train_file", type=str)
    parser.add_argument("test_file", type=str)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--n_inducing_points", type=int, default=100)
    parser.add_argument("--encoder_dim", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1e-3)
    parser.add_argument("--jitter", type=float, default=1e-4)
    parser.add_argument("--hparam_prior", type=float, nargs=2, default=None)
    parser.add_argument("--featurizer_model", type=str, default="ibm/MoLFormer-XL-both-10pct")
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)
    
    # --- New/Modified Logging Arguments ---
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="Logging backend to use.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for TensorBoard logs.")
    parser.add_argument("--wandb_project", type=str, default="grokking_chemistry", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default="experiment", help="A name for this specific W&B run.")

    args = parser.parse_args()
    key = jax.random.PRNGKey(args.random_seed)

    # --- Setup Logger ---
    if args.logger == "wandb":
        logger = WandbLogger(config=vars(args), run_name=args.wandb_run_name, project_name=args.wandb_project)
    else: # tensorboard
        logger = TensorBoardLogger(config=vars(args), run_name=args.wandb_run_name, log_dir=args.log_dir)

    # --- THE FIX: Use a try...finally block to guarantee logger cleanup ---
    try:
        # --- Load and filter data ---
        print("Loading and filtering data...")
        x_train_df = pl.read_csv(args.train_file).filter(pl.col('smiles').is_not_null() & (pl.col('smiles') != ""))
        x_test_df = pl.read_csv(args.test_file).filter(pl.col('smiles').is_not_null() & (pl.col('smiles') != ""))
        print(f"Found {len(x_train_df)} valid training molecules.")
        print(f"Found {len(x_test_df)} valid test molecules.")
        
        # --- Featurize Data ---
        X_train = featurize_smiles(x_train_df['smiles'].to_list(), model_name=args.featurizer_model)
        X_test = featurize_smiles(x_test_df['smiles'].to_list(), model_name=args.featurizer_model)
        y_train = x_train_df['active'].to_numpy()
        y_test = x_test_df['active'].to_numpy()

        # --- Setup and Train Model ---
        callback = log_callback_factory(X_train, y_train, X_test, y_test, logger)
        key, encoder_key, hparam_key = jax.random.split(key, 3)
        encoder = LinearEncoder(encoder_key, X_train.shape[-1], args.encoder_dim)
        
        log_gamma_init = jax.random.normal(hparam_key, shape=(args.encoder_dim,)) if args.encoder_dim > 1 else jax.random.normal(hparam_key, shape=())
        hparams_init = {'log_gamma': log_gamma_init}

        svgpc = SparseVariationalGPClassifier(
            learning_rate=args.learning_rate, n_inducing_points=args.n_inducing_points,
            sampler='sgld', temperature=args.temperature, train_inducing_points=True,
            ard=True, log_callback=callback, jitter=args.jitter, encoder=encoder,
            encoder_params=encoder.params, kernel_hparams_init=hparams_init,
            hparam_prior=args.hparam_prior
        )
        
        print("Starting training...")
        svgpc.fit(X_train, y_train, epochs=args.epochs, batch_size=min(args.batch_size, len(y_train)), random_seed=args.random_seed)
        
        print("Training complete.")

    finally:
        # This block will execute no matter how the 'try' block exits
        print("Finalizing logger...")
        logger.finish()

if __name__ == '__main__':
    main()


