import jax
import jax.numpy as jnp
from functools import partial
import time
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

class JaxOutOfCoreSVM:
    """
    An "out-of-core" version of the JAX Linear SVM.

    This model is designed to handle datasets where the training matrix is too
    large to assemble in memory. It works by taking pre-computed, unique feature
    matrices (e.g., for all unique ligands and all unique proteins) and an
    array of index pairs that define the actual training samples.

    The feature concatenation for each training sample is performed on-the-fly
    inside the JIT-compiled training loop, which is highly efficient.

    This class assumes that the input feature matrices (`X_ligand`, `X_protein`)
    have already been scaled and normalized.
    """
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, random_seed=42, epsilon=1e-5, jitter=1e-6, predict_batch_size=128):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.key = jax.random.PRNGKey(random_seed)
        self.epsilon = epsilon
        self.jitter = jitter
        self.predict_batch_size = predict_batch_size

        # Model state is initialized in fit()
        self.alphas = None
        self.b = None
        self.w = None

    def fit(self, X_ligand, X_protein, train_pairs, y):
        """
        Fits the SVM model.

        Args:
            X_ligand (jnp.ndarray): Pre-scaled feature matrix for unique ligands.
            X_protein (jnp.ndarray): Pre-scaled feature matrix for unique proteins.
            train_pairs (jnp.ndarray): Array of shape (n_samples, 2) where each
                                       row is `(ligand_idx, protein_idx)`.
            y (jnp.ndarray): Target labels {0, 1} or {-1, 1} of shape (n_samples,).
        """
        y_original_for_auc = y
        unique_labels = jnp.unique(y)
        if jnp.array_equal(unique_labels, jnp.array([0, 1])):
            print("Labels detected as {0, 1}. Converting to {-1, 1} for SVM formulation.")
            y = jnp.where(y == 0, -1, 1)
        
        n_samples = train_pairs.shape[0]
        n_features = X_ligand.shape[1] + X_protein.shape[1]

        self.alphas = jnp.zeros(n_samples)
        self.b = 0.0
        self.w = jnp.zeros(n_features)
        
        positive_indices = jnp.where(y == 1)[0]
        negative_indices = jnp.where(y == -1)[0]

        @partial(jax.jit, static_argnames=['n_samples'])
        def run_epoch(carry, indices, X_ligand, X_protein, train_pairs, y, pos_indices, neg_indices, n_samples):
            def update_step(alphas, w, b, i, j):
                lig_i, prot_i = train_pairs[i]
                x_i = jnp.concatenate([X_ligand[lig_i], X_protein[prot_i]])
                lig_j, prot_j = train_pairs[j]
                x_j = jnp.concatenate([X_ligand[lig_j], X_protein[prot_j]])
                alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                error_i = jnp.dot(w, x_i) + b - y[i]
                error_j = jnp.dot(w, x_j) + b - y[j]
                L, H = jax.lax.cond(y[i] != y[j], lambda: (jnp.maximum(0, alpha_j_old - alpha_i_old), jnp.minimum(self.C, self.C + alpha_j_old - alpha_i_old)), lambda: (jnp.maximum(0, alpha_i_old + alpha_j_old - self.C), jnp.minimum(self.C, alpha_i_old + alpha_j_old)))
                k_ii, k_jj, k_ij = jnp.dot(x_i, x_i), jnp.dot(x_j, x_j), jnp.dot(x_i, x_j)
                eta = 2.0 * k_ij - k_ii - k_jj - self.jitter
                def optimization_logic():
                    alpha_j_new = alpha_j_old - (y[j] * (error_i - error_j)) / eta
                    alpha_j_new_clipped = jnp.clip(alpha_j_new, L, H)
                    delta_alpha_j = alpha_j_new_clipped - alpha_j_old
                    def update_and_report_change():
                        alpha_i_new = alpha_i_old - y[i] * y[j] * delta_alpha_j
                        new_alphas = alphas.at[i].set(alpha_i_new).at[j].set(alpha_j_new_clipped)
                        b1 = b - error_i - y[i] * (alpha_i_new - alpha_i_old) * k_ii - y[j] * delta_alpha_j * k_ij
                        b2 = b - error_j - y[i] * (alpha_i_new - alpha_i_old) * k_ij - y[j] * delta_alpha_j * k_jj
                        cond1 = (0 < alpha_i_new) & (alpha_i_new < self.C); cond2 = (0 < alpha_j_new_clipped) & (alpha_j_new_clipped < self.C)
                        new_b = jax.lax.cond(cond1, lambda: b1, lambda: jax.lax.cond(cond2, lambda: b2, lambda: (b1 + b2) / 2.0))
                        new_w = w + (alpha_i_new - alpha_i_old) * y[i] * x_i + delta_alpha_j * y[j] * x_j
                        return new_alphas, new_w, new_b, True
                    return jax.lax.cond(jnp.abs(delta_alpha_j) > self.epsilon, update_and_report_change, lambda: (alphas, w, b, False))
                return jax.lax.cond((H - L > self.epsilon) & (eta < 0), optimization_logic, lambda: (alphas, w, b, False))
            def epoch_body(carry, i):
                alphas, w, b, key, count = carry
                lig_idx_i, prot_idx_i = train_pairs[i]; x_i = jnp.concatenate([X_ligand[lig_idx_i], X_protein[prot_idx_i]])
                error_i = jnp.dot(w, x_i) + b - y[i]
                kkt_violated = ((y[i] * error_i < -self.tol) & (alphas[i] < self.C)) | ((y[i] * error_i > self.tol) & (alphas[i] > 0))
                def perform_update():
                    new_key, subkey = jax.random.split(key)
                    def pick_opposite(): return jax.lax.cond(y[i] == 1, lambda: jax.random.choice(subkey, neg_indices), lambda: jax.random.choice(subkey, pos_indices))
                    def pick_any(): rand_j = jax.random.choice(subkey, n_samples); return jnp.where(rand_j == i, (rand_j + 1) % n_samples, rand_j)
                    j = jax.lax.cond((pos_indices.shape[0] > 0) & (neg_indices.shape[0] > 0), pick_opposite, pick_any)
                    new_alphas, new_w, new_b, changed = update_step(alphas, w, b, i, j)
                    return new_alphas, new_w, new_b, new_key, count + jnp.where(changed, 1, 0)
                return jax.lax.cond(kkt_violated, perform_update, lambda: carry), None
            final_carry, _ = jax.lax.scan(epoch_body, carry, indices); return final_carry

        print("Starting Out-of-Core SVM training with JAX (using lax.scan)...")
        for iter_num in range(self.max_iter):
            self.key, subkey = jax.random.split(self.key)
            shuffled_indices = jax.random.permutation(subkey, n_samples)
            initial_carry = (self.alphas, self.w, self.b, self.key, 0)
            final_carry = run_epoch(initial_carry, shuffled_indices, X_ligand, X_protein, train_pairs, y, positive_indices, negative_indices, n_samples)
            self.alphas, self.w, self.b, self.key, alphas_changed = final_carry

            print(f"Iteration {iter_num + 1}/{self.max_iter} | Alpha pairs changed: {alphas_changed}")
            if alphas_changed == 0 and iter_num > 0:
                print("Convergence reached."); break
        print("Training finished.")

    def decision_function(self, X_ligand, X_protein, test_pairs, batch_size=None):
        """
        Computes the decision function in batches to avoid OOM errors.
        """
        if batch_size is None:
            batch_size = self.predict_batch_size
        
        n_test = test_pairs.shape[0]
        all_scores = []

        @jax.jit
        def predict_batch(X_ligand, X_protein, batch_pairs):
            lig_vecs = X_ligand.at[batch_pairs[:, 0]].get()
            prot_vecs = X_protein.at[batch_pairs[:, 1]].get()
            
            @jax.jit
            def predict_single(lig_vec, prot_vec):
                features = jnp.concatenate([lig_vec, prot_vec])
                return jnp.dot(self.w, features) + self.b
            return jax.vmap(predict_single)(lig_vecs, prot_vecs)

        for i in tqdm(range(0, n_test, batch_size), desc="Predicting in batches"):
            batch_pairs = test_pairs[i:i+batch_size]
            batch_scores = predict_batch(X_ligand, X_protein, batch_pairs)
            all_scores.append(batch_scores)

        return jnp.concatenate(all_scores)

    def predict(self, X_ligand, X_protein, test_pairs, batch_size=None):
        """Predicts class labels in batches."""
        scores = self.decision_function(X_ligand, X_protein, test_pairs, batch_size)
        return jnp.sign(scores)


