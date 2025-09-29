import jax
import jax.numpy as jnp
from functools import partial
import time
from tqdm.auto import tqdm

class JaxOutOfCoreKernelSVM:
    """
    An "out-of-core" Kernel SVM with a robust training loop that shuffles
    data and runs for a full number of epochs to ensure better convergence.
    """
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, random_seed=42, 
                 epsilon=1e-5, jitter=1e-6, predict_batch_size=128,
                 kernel='rbf', gamma=0.01):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.key = jax.random.PRNGKey(random_seed)
        self.epsilon = epsilon
        self.jitter = jitter
        self.predict_batch_size = predict_batch_size

        self.kernel_name = kernel
        self.gamma = gamma
        self.kernel = self._get_kernel_function()

        self.alphas = None
        self.b = 0.0
        self.train_pairs = None
        self.y_train = None

    def _get_kernel_function(self):
        """Factory to get the desired kernel function."""
        if self.kernel_name == 'rbf':
            def rbf_kernel(x1, x2):
                return jnp.exp(-self.gamma * jnp.sum((x1 - x2)**2))
            return rbf_kernel
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_name}")

    def fit(self, X_ligand, X_protein, train_pairs, y):
        """
        Fits the Kernel SVM model.
        """
        if jnp.array_equal(jnp.unique(y), jnp.array([0, 1])):
            print("Labels detected as {0, 1}. Converting to {-1, 1} for SVM formulation.")
            y = jnp.where(y == 0, -1, 1)
        
        self.train_pairs = train_pairs
        self.y_train = y
        n_samples = train_pairs.shape[0]

        self.alphas = jnp.zeros(n_samples)
        self.b = 0.0
        errors = -y.astype(float)

        @jax.jit
        def run_epoch(carry, indices, x_lig, x_prot, pairs_data, y_data):
            jitted_epoch_body_fn = self._make_epoch_body(x_lig, x_prot, pairs_data, y_data)
            final_carry, _ = jax.lax.scan(jitted_epoch_body_fn, carry, indices)
            return final_carry

        print(f"Starting Out-of-Core Kernel SVM training (kernel: {self.kernel_name})...")
        for iter_num in range(self.max_iter):
            self.key, subkey = jax.random.split(self.key)
            shuffled_indices = jax.random.permutation(subkey, n_samples)
            
            initial_carry = (self.alphas, self.b, errors, 0)
            
            final_carry = run_epoch(initial_carry, shuffled_indices, X_ligand, X_protein, train_pairs, y)
            
            self.alphas, self.b, errors, alphas_changed = final_carry

            print(f"Iteration {iter_num + 1}/{self.max_iter} | Alpha pairs changed: {alphas_changed}")
            
            # --- FIX: Removed the premature early stopping condition ---
            # We now let the training run for the full number of epochs to allow
            # shuffling to help the optimizer find the true global minimum.
            
        print("Training finished.")

    def _make_epoch_body(self, X_lig, X_prot, pairs, y):
        update_step_fn = self._make_update_step(X_lig, X_prot, pairs, y)

        def epoch_body(carry, i):
            alphas, b, errors, count = carry
            error_i = errors[i]
            kkt_violates = ((y[i] * error_i < -self.tol) & (alphas[i] < self.C)) | \
                           ((y[i] * error_i > self.tol) & (alphas[i] > 0))

            def perform_update(operands):
                a, b_op, err, c = operands
                j = jnp.argmax(jnp.abs(error_i - err).at[i].set(-jnp.inf))
                new_a, new_b, new_err, changed = update_step_fn(a, b_op, err, i, j)
                return new_a, new_b, new_err, c + jnp.where(changed, 1, 0)

            return jax.lax.cond(kkt_violates, perform_update, lambda op: op, carry), None
        
        return epoch_body

    def _make_update_step(self, X_lig, X_prot, pairs, y):
        # This function remains unchanged
        def update_step(alphas, b, errors, i, j):
            x_i = jnp.concatenate([X_lig[pairs[i, 0]], X_prot[pairs[i, 1]]])
            x_j = jnp.concatenate([X_lig[pairs[j, 0]], X_prot[pairs[j, 1]]])

            alpha_i_old, alpha_j_old = alphas[i], alphas[j]
            error_i, error_j = errors[i], errors[j]

            L, H = jax.lax.cond(y[i] != y[j],
                lambda: (jnp.maximum(0, alpha_j_old - alpha_i_old), jnp.minimum(self.C, self.C + alpha_j_old - alpha_i_old)),
                lambda: (jnp.maximum(0, alpha_i_old + alpha_j_old - self.C), jnp.minimum(self.C, alpha_i_old + alpha_j_old)))

            k_ii = self.kernel(x_i, x_i); k_jj = self.kernel(x_j, x_j); k_ij = self.kernel(x_i, x_j)
            eta = 2.0 * k_ij - k_ii - k_jj - self.jitter

            def optimization_logic(_):
                alpha_j_new = alpha_j_old - (y[j] * (error_i - error_j)) / eta
                alpha_j_new_clipped = jnp.clip(alpha_j_new, L, H)
                delta_alpha_j = alpha_j_new_clipped - alpha_j_old

                def update_and_report(_):
                    alpha_i_new = alpha_i_old - y[i] * y[j] * delta_alpha_j
                    delta_alpha_i = alpha_i_new - alpha_i_old
                    new_alphas = alphas.at[i].set(alpha_i_new).at[j].set(alpha_j_new_clipped)
                    
                    b1 = b - error_i - y[i] * delta_alpha_i * k_ii - y[j] * delta_alpha_j * k_ij
                    b2 = b - error_j - y[i] * delta_alpha_i * k_ij - y[j] * delta_alpha_j * k_jj
                    cond1 = jnp.logical_and(0 < alpha_i_new, alpha_i_new < self.C)
                    cond2 = jnp.logical_and(0 < alpha_j_new_clipped, alpha_j_new_clipped < self.C)
                    new_b = jax.lax.cond(cond1, lambda: b1, lambda: jax.lax.cond(cond2, lambda: b2, lambda: (b1 + b2) / 2.0))
                    
                    @jax.vmap
                    def kernel_vmap(idx):
                        x_k = jnp.concatenate([X_lig[pairs[idx, 0]], X_prot[pairs[idx, 1]]])
                        k_ik = self.kernel(x_i, x_k); k_jk = self.kernel(x_j, x_k)
                        return errors[idx] + delta_alpha_i*y[i]*k_ik + delta_alpha_j*y[j]*k_jk + (new_b - b)

                    new_errors = kernel_vmap(jnp.arange(len(y)))
                    return new_alphas, new_b, new_errors, True

                return jax.lax.cond(jnp.abs(delta_alpha_j) > self.epsilon, update_and_report, lambda _: (alphas, b, errors, False), None)

            return jax.lax.cond((H - L > self.epsilon) & (eta < 0), optimization_logic, lambda _: (alphas, b, errors, False), None)
        return update_step

    def decision_function(self, X_ligand, X_protein, test_pairs):
        # This function remains unchanged
        n_test = test_pairs.shape[0]
        all_scores = jnp.zeros(n_test)
        sv_indices = jnp.where(self.alphas > self.epsilon)[0]
        sv_alphas = self.alphas[sv_indices]
        sv_y = self.y_train[sv_indices]
        sv_pairs = self.train_pairs[sv_indices]
        
        print(f"Making predictions using {len(sv_indices)} support vectors.")

        @jax.jit
        def predict_batch(x_lig, x_prot, batch_test_pairs):
            def single_prediction(test_pair):
                x_test = jnp.concatenate([x_lig[test_pair[0]], x_prot[test_pair[1]]])
                
                @jax.vmap
                def kernel_vmap(sv_pair):
                    x_sv = jnp.concatenate([x_lig[sv_pair[0]], x_prot[sv_pair[1]]])
                    return self.kernel(x_test, x_sv)
                
                kernel_values = kernel_vmap(sv_pairs)
                return jnp.dot(sv_alphas * sv_y, kernel_values) + self.b

            return jax.vmap(single_prediction)(batch_test_pairs)

        for i in tqdm(range(0, n_test, self.predict_batch_size), desc="Predicting in batches"):
            batch_pairs = test_pairs[i:i+self.predict_batch_size]
            batch_scores = predict_batch(X_ligand, X_protein, batch_pairs)
            all_scores = all_scores.at[i:i+len(batch_scores)].set(batch_scores)

        return all_scores

    def predict(self, X_ligand, X_protein, test_pairs):
        scores = self.decision_function(X_ligand, X_protein, test_pairs)
        return jnp.sign(scores)
