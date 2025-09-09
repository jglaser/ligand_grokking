import argparse
import numpy as np
import polars as pl
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Local import of your classifier
from svgp_classifier import SparseVariationalGPClassifier

# --- Encoder Definition ---
class LinearEncoder:
    """A simple linear encoder with a single weight matrix."""
    def __init__(self, key, input_dim, output_dim):
        # Initialize weights with variance scaling for stability
        self.params = {'W': jax.random.normal(key, shape=(input_dim, output_dim)) * jnp.sqrt(2.0 / input_dim)}
        self.output_dim = output_dim

    def __call__(self, params, x):
        return x @ params['W']

# --- HTSR Alpha Calculation (Robust MLE Method) ---
def estimate_alpha_fit(data):
    """
    Estimates the power-law exponent 'alpha' of a distribution's tail
    using the Maximum Likelihood Estimator (MLE) method from Clauset et al., 2009.
    This version is more robust to small sample sizes.
    """
    data = np.asarray(data)
    data = data[data > 1e-9] # Filter out numerical zeros and negatives
    # Lowered the threshold to handle small numbers of singular values
    if len(data) < 4:
        return np.nan

    sorted_data = np.sort(data)
    xmins = np.unique(sorted_data)
    
    # Ensure we don't try to select a tail from a single point
    if len(xmins) > 1:
        xmins = xmins[:-1]

    best_ks = np.inf
    best_alpha = np.nan

    for xmin in xmins:
        if xmin <= 0: continue
        tail = sorted_data[sorted_data >= xmin]
        n_tail = len(tail)
        if n_tail < 2:
            continue

        # MLE for alpha, with added epsilon for stability
        log_sum = np.sum(np.log(tail / xmin))
        alpha = 1.0 + n_tail / (log_sum + 1e-9)

        # KS statistic
        empirical_cdf = np.arange(n_tail) / n_tail
        fitted_cdf = 1 - (tail / xmin)**(-alpha + 1)
        ks_stat = np.max(np.abs(empirical_cdf - fitted_cdf))

        if ks_stat < best_ks:
            best_ks = ks_stat
            best_alpha = alpha

    return best_alpha


def featurize_smiles(smiles_list, model_name="ibm/MoLFormer-XL-both-10pct", batch_size=64, max_length=512):
    """Generates feature embeddings for a list of SMILES strings."""
    print(f"Loading feature extractor model: {model_name}...")
    # Using device=0 assumes a CUDA-enabled GPU is available at index 0.
    # Change to device=-1 for CPU.
    # --- Featurization ---
    from transformers import pipeline
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    feature_extractor = pipeline(
        "feature-extraction",
        tokenizer=tokenizer,
        framework="pt",
        model=model_name,
        device=0,
        trust_remote_code=True
    )
    print("Model loaded. Starting featurization...")

    all_features = []
    # Process the list in batches to manage memory
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Featurizing Batches"):
        batch_smiles = [s[:max_length] for s in smiles_list[i:i + batch_size]]
        
        # The pipeline returns a list of tensors, one for each SMILES
        outputs = feature_extractor(batch_smiles, return_tensors="pt")
        
        # We take the mean of the token embeddings for each molecule to get a fixed-size vector
        batch_features = np.stack([t[0].numpy().mean(axis=0) for t in outputs])
        all_features.append(batch_features)

    return np.vstack(all_features)

# --- Logging Callback ---
def log_callback_factory(X_train, y_train, X_val, y_val):
    """
    Creates a callback function for logging metrics during training.
    This factory pattern allows us to pass training and validation data to the callback.
    """
    def log_callback(model, epoch, metrics, params, bias_state):
        # Log accuracy every 100 epochs
        if epoch % 100 == 0:
            val_score = model.score(X_val, y_val, params=params)
            metrics["validation_accuracy"] = val_score

            # Calculate training accuracy on a random subset for speed
            train_subset_idx = np.random.choice(X_train.shape[0], size=min(len(y_val), len(y_train)), replace=False)
            X_train_subset = X_train[train_subset_idx]
            y_train_subset = y_train[train_subset_idx]
            train_score = model.score(X_train_subset, y_train_subset, params=params)
            metrics["train_accuracy"] = train_score
            
        # Log HTSR alpha every 10 epochs
        if epoch % 10 == 0 and 'encoder_params' in params and 'W' in params['encoder_params']:
            svals = jnp.linalg.svd(params['encoder_params']['W'], compute_uv=False)
            metrics['htsr_alpha'] = estimate_alpha_fit(svals**2)
            
        return metrics
    return log_callback

def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train an SVGP classifier with a trainable encoder on molecular data.")
    parser.add_argument("train_file", type=str, help="Path to the training CSV file.")
    parser.add_argument("test_file", type=str, help="Path to the test CSV file.")
    
    # Model Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--n_inducing_points", type=int, default=100)
    parser.add_argument("--encoder_dim", type=int, default=8, help="Dimensionality of the learned representation.")
    parser.add_argument("--temperature", type=float, default=1e-3, help="Temperature for SGLD sampling.")
    parser.add_argument("--jitter", type=float, default=1e-4)
    parser.add_argument("--hparam_prior", type=float, nargs=2, default=None, help="Shape and rate for the Gamma prior on kernel hparams (e.g., 2.0 2.0).")

    # Featurization settings
    parser.add_argument("--featurizer_model", type=str, default="ibm/MoLFormer-XL-both-10pct", help="Hugging Face model name for featurization.")

    # Training settings
    parser.add_argument("--epochs", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)

    # W&B Logging
    parser.add_argument("--wandb_project", type=str, default="grokking_chemistry", help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default="experiment", help="A name for this specific W&B run.")

    args = parser.parse_args()
    key = jax.random.PRNGKey(args.random_seed)

    # --- 1. Load Data & Add Robust Filtering ---
    print("Loading and filtering data...")
    x_train_df = pl.read_csv(args.train_file)
    x_test_df = pl.read_csv(args.test_file)

    # Filter out rows with null or empty SMILES strings to prevent errors
    x_train_df = x_train_df.filter(pl.col('smiles').is_not_null() & (pl.col('smiles') != ""))
    x_test_df = x_test_df.filter(pl.col('smiles').is_not_null() & (pl.col('smiles') != ""))
    
    print(f"Found {len(x_train_df)} valid training molecules.")
    print(f"Found {len(x_test_df)} valid test molecules.")


    # --- 2. Featurize Data ---
    X_train = featurize_smiles(x_train_df['smiles'].to_list(), model_name=args.featurizer_model)
    X_test = featurize_smiles(x_test_df['smiles'].to_list(), model_name=args.featurizer_model)
    y_train = x_train_df['active'].to_numpy()
    y_test = x_test_df['active'].to_numpy()

    # --- 3. Setup and Train Model ---
    print("Setting up SVGP classifier...")
    run = None
    if WANDB_AVAILABLE:
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Create the callback function with access to the full train and test sets
    callback = log_callback_factory(X_train, y_train, X_test, y_test)
    
    # Initialize the trainable encoder
    key, encoder_key, hparam_key = jax.random.split(key, 3)
    encoder = LinearEncoder(encoder_key, X_train.shape[-1], args.encoder_dim)

    # Initialize kernel hyperparameters from a normal distribution
    if args.encoder_dim > 1:
        log_gamma_init = jax.random.normal(hparam_key, shape=(args.encoder_dim,))
    else:
        log_gamma_init = jax.random.normal(hparam_key, shape=())
    hparams_init = {'log_gamma': log_gamma_init}

    svgpc = SparseVariationalGPClassifier(
        learning_rate=args.learning_rate,
        n_inducing_points=args.n_inducing_points,
        sampler='sgld',
        temperature=args.temperature,
        train_inducing_points=True,
        ard=True,
        log_callback=callback,
        jitter=args.jitter,
        encoder=encoder,
        encoder_params=encoder.params,
        kernel_hparams_init=hparams_init,
        hparam_prior=args.hparam_prior
    )

    print("Starting training...")
    svgpc.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=min(args.batch_size, len(y_train)),
        random_seed=args.random_seed,
        wandb_run=run
    )

    print("Training complete.")
    if run:
        run.finish()

if __name__ == '__main__':
    main()


