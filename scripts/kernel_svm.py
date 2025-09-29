import jax
import jax.numpy as jnp
from functools import partial
import time # For benchmarking

from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Imports for scikit-learn benchmark
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline


class JaxKernelSVM:
    """
    A final, corrected Kernel SVM with proper JIT compilation patterns for
    prediction, resolving the non-hashable arguments error.
    """
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, random_seed=42,
                 epsilon=1e-5, jitter=1e-6, kernel='rbf', gamma='scale'):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.key = jax.random.PRNGKey(random_seed)
        self.epsilon = epsilon
        self.jitter = jitter
        self.gamma_val = gamma 

        self.alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.mean_ = None
        self.scale_ = None
        self.kernel = None
        self.kernel_name = kernel


    def _get_kernel_function(self, n_features):
        if self.gamma_val == 'scale':
            gamma = 1.0 / (n_features * self.X_train.var())
        elif self.gamma_val == 'auto':
            gamma = 1.0 / n_features
        else:
            gamma = self.gamma_val

        if self.kernel_name == 'rbf':
            return lambda x1, x2: jnp.exp(-gamma * jnp.sum((x1 - x2)**2))
        elif self.kernel_name == 'linear':
            return lambda x1, x2: jnp.dot(x1, x2)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_name}")
            
    def _preprocess(self, X, fit_transform=False):
        if fit_transform:
            self.mean_ = jnp.mean(X, axis=0)
            self.scale_ = jnp.std(X, axis=0)
        
        safe_scale = jnp.where(self.scale_ == 0, 1.0, self.scale_)
        X_scaled = (X - self.mean_) / safe_scale
        
        norms = jnp.linalg.norm(X_scaled, axis=1, keepdims=True)
        X_normalized = X_scaled / (norms + self.epsilon)
        
        return jnp.nan_to_num(X_normalized)


    def fit(self, X, y):
        unique_labels = jnp.unique(y)
        if jnp.all(jnp.sort(unique_labels) == jnp.array([0, 1])):
            print("Labels detected in {0, 1} format. Converting to {-1, 1} for SVM training.")
            y = jnp.where(y == 0, -1, 1)

        self.X_train = self._preprocess(X, fit_transform=True)
        self.y_train = y
        n_samples, n_features = self.X_train.shape
        self.kernel = self._get_kernel_function(n_features)

        self.alphas = jnp.zeros(n_samples)
        self.b = 0.0
        errors = -y.astype(float)

        epoch_body_fn = self._make_epoch_body(self.X_train, self.y_train)

        @jax.jit
        def run_epoch(carry, indices):
            final_carry, _ = jax.lax.scan(epoch_body_fn, carry, indices)
            return final_carry

        print(f"Starting JIT-compiled Kernel SVM training (kernel: {self.kernel_name})...")
        for iter_num in range(self.max_iter):
            initial_carry = (self.alphas, self.b, errors, 0)
            self.key, subkey = jax.random.split(self.key)
            shuffled_indices = jax.random.permutation(subkey, n_samples)
            
            final_carry = run_epoch(initial_carry, shuffled_indices)
            self.alphas, self.b, errors, alphas_changed = final_carry
            
            print(f"Iteration {iter_num + 1}/{self.max_iter} | Alpha pairs changed: {alphas_changed}")
            if alphas_changed == 0 and iter_num > 1:
                print("Convergence reached.")
                break
        print("Training finished.")

    def decision_function(self, X):
        """Computes the decision function for a given set of samples."""
        X_processed = self._preprocess(X, fit_transform=False)

        # --- FIX: Define a static helper function to JIT ---
        # This function does not close over 'self' and only takes JAX arrays.
        @jax.jit
        def batch_predict(alphas, y_train, b, x_train, x_test_batch):
            def single_prediction(x_test):
                kernel_values = jax.vmap(self.kernel, in_axes=(0, None))(x_train, x_test)
                return jnp.dot(alphas * y_train, kernel_values) + b
            return jax.vmap(single_prediction)(x_test_batch)

        # Call the JIT'd function with the necessary data from the class
        return batch_predict(self.alphas, self.y_train, self.b, self.X_train, X_processed)

    def predict(self, X):
        """Predicts the class labels for a given set of samples."""
        # The decision_function is now safe to call, as the JIT is handled internally
        return jnp.sign(self.decision_function(X))


    def _make_epoch_body(self, X, y):
        update_step_fn = self._make_update_step(X, y)

        def epoch_body(carry, i):
            alphas, b, errors, alphas_changed_count = carry
            error_i = errors[i]
            
            kkt_violated = ((y[i] * error_i < -self.tol) & (alphas[i] < self.C)) | \
                           ((y[i] * error_i > self.tol) & (alphas[i] > 0))

            def perform_update(operands):
                alphas_op, b_op, errors_op, count_op = operands
                error_diffs = jnp.abs(error_i - errors_op)
                j = jnp.argmax(error_diffs.at[i].set(-jnp.inf))
                
                new_alphas, new_b, new_errors, changed = update_step_fn(alphas_op, b_op, errors_op, i, j)
                new_count = count_op + jnp.where(changed, 1, 0)
                return new_alphas, new_b, new_errors, new_count

            def no_update(operands):
                return operands
            
            new_carry = jax.lax.cond(kkt_violated, perform_update, no_update, 
                                     (alphas, b, errors, alphas_changed_count))
            return new_carry, None
        return epoch_body

    def _make_update_step(self, X, y):
        def update_step(alphas, b, errors, i, j):
            alpha_i_old, alpha_j_old = alphas[i], alphas[j]
            error_i, error_j = errors[i], errors[j]

            L, H = jax.lax.cond(
                y[i] != y[j],
                lambda: (jnp.maximum(0, alpha_j_old - alpha_i_old), jnp.minimum(self.C, self.C + alpha_j_old - alpha_i_old)),
                lambda: (jnp.maximum(0, alpha_i_old + alpha_j_old - self.C), jnp.minimum(self.C, alpha_i_old + alpha_j_old))
            )

            k_ii = self.kernel(X[i], X[i]); k_jj = self.kernel(X[j], X[j]); k_ij = self.kernel(X[i], X[j])
            eta = 2.0 * k_ij - k_ii - k_jj - self.jitter

            def optimization_logic(_):
                alpha_j_new = alpha_j_old - (y[j] * (error_i - error_j)) / eta
                alpha_j_new_clipped = jnp.clip(alpha_j_new, L, H)
                delta_alpha_j = alpha_j_new_clipped - alpha_j_old

                def update_and_report_change(_):
                    alpha_i_new = alpha_i_old - y[i] * y[j] * delta_alpha_j
                    new_alphas = alphas.at[i].set(alpha_i_new).at[j].set(alpha_j_new_clipped)
                    delta_alpha_i = alpha_i_new - alpha_i_old
                    
                    b1 = b - error_i - y[i] * delta_alpha_i * k_ii - y[j] * delta_alpha_j * k_ij
                    b2 = b - error_j - y[i] * delta_alpha_i * k_ij - y[j] * delta_alpha_j * k_jj
                    cond1 = jnp.logical_and(0 < alpha_i_new, alpha_i_new < self.C)
                    cond2 = jnp.logical_and(0 < alpha_j_new_clipped, alpha_j_new_clipped < self.C)
                    new_b = jax.lax.cond(cond1, lambda: b1, lambda: jax.lax.cond(cond2, lambda: b2, lambda: (b1 + b2) / 2.0))
                    delta_b = new_b - b

                    kernel_col_i = jax.vmap(self.kernel, in_axes=(0, None))(X, X[i])
                    kernel_col_j = jax.vmap(self.kernel, in_axes=(0, None))(X, X[j])
                    new_errors = errors + delta_alpha_i * y[i] * kernel_col_i + delta_alpha_j * y[j] * kernel_col_j + delta_b
                    return new_alphas, new_b, new_errors, True

                def no_change_and_report(_):
                    return alphas, b, errors, False

                return jax.lax.cond(jnp.abs(delta_alpha_j) > self.epsilon, update_and_report_change, no_change_and_report, None)

            def no_op(_):
                return alphas, b, errors, False

            return jax.lax.cond((H - L > self.epsilon) & (eta < 0), optimization_logic, no_op, None)
        return update_step


if __name__ == '__main__':
    print(f"\n{'='*20} BENCHMARKING KERNEL SVM {'='*20}")
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    C_param = 5.0
    gamma_param = 'scale' 

    print("\n--- Training JAX Kernel SVM (RBF Kernel) ---")
    svm = JaxKernelSVM(C=C_param, max_iter=50, gamma=gamma_param, random_seed=42)
    X_train_jax, y_train_jax = jnp.array(X_train), jnp.array(y_train)

    start_time_jax = time.perf_counter()
    svm.fit(X_train_jax, y_train_jax)
    end_time_jax = time.perf_counter()
    jax_train_time = end_time_jax - start_time_jax

    y_pred_jax = svm.predict(jnp.array(X_test))
    accuracy_jax = accuracy_score(jnp.where(y_test==0,-1,1), y_pred_jax)

    print("\n--- Training scikit-learn SVC (RBF Kernel) ---")
    sklearn_svc = Pipeline([
        ('scaler', StandardScaler()),
        ('normalizer', Normalizer()),
        ('svc', SVC(C=C_param, gamma=gamma_param, kernel='rbf', tol=1e-4))
    ])

    start_time_sklearn = time.perf_counter()
    sklearn_svc.fit(X_train, y_train)
    end_time_sklearn = time.perf_counter()
    y_pred_sklearn = sklearn_svc.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print("\n\n" + "="*70)
    print("--- FINAL BENCHMARK RESULTS (Matched Preprocessing & Correct JIT) ---")
    print("="*70)
    print(f"{'Implementation':>15} | {'Train Time (s)':>18} | {'Accuracy (%)':>15}")
    print("-"*70)
    print(f"{'JAX Kernel SVM':>15} | {jax_train_time:>18.4f} | {accuracy_jax*100:>15.2f}")
    print(f"{'Scikit-learn SVC':>15} | {end_time_sklearn - start_time_sklearn:>18.4f} | {accuracy_sklearn*100:>15.2f}")
    print("="*70)
