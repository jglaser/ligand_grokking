import jax
import jax.numpy as jnp
from functools import partial

# This function no longer needs the faulty JIT decorator.
# The performance gain from JITing this simple update is negligible
# compared to the data loading, and removing it fixes the error.
def _update_scaler_stats(mean, var, n, batch):
    """
    A pure JAX function to perform a stable, one-pass update of mean and variance.
    Uses a parallel variance algorithm.
    """
    batch_n, n_features = batch.shape
    new_n = n + batch_n
    
    # Update mean
    batch_mean = jnp.mean(batch, axis=0)
    delta = batch_mean - mean
    new_mean = mean + delta * (batch_n / new_n)

    # Update variance
    batch_var = jnp.var(batch, axis=0)
    m_a = var * n
    m_b = batch_var * batch_n
    m2 = m_a + m_b + jnp.square(delta) * n * batch_n / new_n
    new_var = m2 / new_n
    
    return new_mean, new_var, new_n

class BatchedStandardScaler:
    """
    A JAX-based standard scaler that can be fit incrementally on batches of data,
    making it suitable for out-of-core learning where the full dataset cannot
    be held in memory.
    """
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        self.n_features_ = None

    def partial_fit(self, X_batch):
        """
        Updates the running mean and variance based on a new batch of data.
        """
        if self.mean_ is None:
            # First batch initialization
            self.n_features_ = X_batch.shape[1]
            self.mean_ = jnp.zeros(self.n_features_, dtype=jnp.float32)
            self.var_ = jnp.zeros(self.n_features_, dtype=jnp.float32)
            self.n_samples_seen_ = 0

        self.mean_, self.var_, self.n_samples_seen_ = _update_scaler_stats(
            self.mean_, self.var_, self.n_samples_seen_, X_batch
        )

    def partial_fit_indexed(self, X_ligand, X_protein, batch_pairs):
        """
        An indexed version of partial_fit that concatenates features on the fly.
        This is much more memory-efficient than pre-concatenating.
        """
        # Assemble the batch from the unique feature matrices using the index pairs
        ligand_batch = X_ligand.at[batch_pairs[:, 0]].get()
        protein_batch = X_protein.at[batch_pairs[:, 1]].get()
        X_batch = jnp.concatenate([ligand_batch, protein_batch], axis=1)

        # Call the original partial_fit with the assembled batch
        self.partial_fit(X_batch)

    def transform(self, X, epsilon=1e-7):
        """
        Scales the data using the learned mean and variance.
        """
        if self.mean_ is None:
            raise RuntimeError("Scaler has not been fit yet. Call partial_fit first.")
        
        # Calculate scale, ensuring we don't divide by zero
        scale = jnp.sqrt(self.var_ + epsilon)
        safe_scale = jnp.where(scale == 0, 1.0, scale)
        
        return (X - self.mean_) / safe_scale


