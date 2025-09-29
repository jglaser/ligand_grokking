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
    A Kernel SVM with a robust, JAX-idiomatic fallback strategy
    to prevent premature convergence.
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
            self.key, subkey = jax.random.split(self.key)
            # Pass the PRNGKey into the scan's carry state for use in fallbacks
            initial_carry = (self.alphas, self.b, errors, 0, subkey)
            shuffled_indices = jax.random.permutation(subkey, n_samples)
            
            final_carry = run_epoch(initial_carry, shuffled_indices)
            self.alphas, self.b, errors, alphas_changed, _ = final_carry
            
            print(f"Iteration {iter_num + 1}/{self.max_iter} | Alpha pairs changed: {alphas_changed}")
            if alphas_changed == 0 and iter_num > 1:
                print("Convergence reached.")
                break
        print("Training finished.")

    def decision_function(self, X):
        X_processed = self._preprocess(X, fit_transform=False)
        @jax.jit
        def batch_predict(alphas, y_train, b, x_train, x_test_batch):
            def single_prediction(x_test):
                kernel_values = jax.vmap(self.kernel, in_axes=(0, None))(x_train, x_test)
                return jnp.dot(alphas * y_train, kernel_values) + b
            return jax.vmap(single_prediction)(x_test_batch)
        return batch_predict(self.alphas, self.y_train, self.b, self.X_train, X_processed)

    def predict(self, X):
        return jnp.sign(self.decision_function(X))

    def _make_search_loop(self, X, y, update_step_fn):
        """Creates a JIT-compatible search loop over candidate indices for j."""
        def search_for_j(carry, candidate_indices):
            alphas, b, errors, i = carry

            def loop_cond(state):
                # State: (alphas, b, errors, changed_flag, search_idx)
                _, _, _, changed, search_idx = state
                return jnp.logical_and(jnp.logical_not(changed), search_idx < candidate_indices.shape[0])

            # In _make_search_loop
            def loop_body(state):
                alphas_s, b_s, errors_s, _, search_idx = state
                j = candidate_indices[search_idx]

                def do_update(_):
                    # This is the original update logic
                    return update_step_fn(alphas_s, b_s, errors_s, i, j)

                def do_nothing(_):
                    # If j is -1, do nothing and report no change
                    return alphas_s, b_s, errors_s, False

                # Only run the update if the index 'j' is valid
                new_alphas, new_b, new_errors, changed = jax.lax.cond(
                    j != -1,
                    do_update,
                    do_nothing,
                    None # Operand is not needed here
                )

                # Conditionally update state: if no change, carry old state forward
                final_alphas = jax.lax.cond(changed, lambda: new_alphas, lambda: alphas_s)
                final_b = jax.lax.cond(changed, lambda: new_b, lambda: b_s)
                final_errors = jax.lax.cond(changed, lambda: new_errors, lambda: errors_s)

                return (final_alphas, final_b, final_errors, changed, search_idx + 1)

            # Initialize state for the while_loop
            init_state = (alphas, b, errors, False, 0)
            final_alphas, final_b, final_errors, changed, _ = jax.lax.while_loop(loop_cond, loop_body, init_state)
            return final_alphas, final_b, final_errors, changed
        
        return search_for_j

    def _make_epoch_body(self, X, y):
        n_samples = X.shape[0]
        update_step_fn = self._make_update_step(X, y)
        search_loop_fn = self._make_search_loop(X, y, update_step_fn)

        def epoch_body(carry, i):
            alphas, b, errors, alphas_changed_count, key = carry
            error_i = errors[i]
            
            kkt_violated = ((y[i] * error_i < -self.tol) & (alphas[i] < self.C)) | \
                           ((y[i] * error_i > self.tol) & (alphas[i] > 0))

            def perform_update(operands):
                alphas_op, b_op, errors_op, count_op, key_op = operands

                # === HEURISTIC 1: Argmax ===
                error_diffs = jnp.abs(error_i - errors_op)
                j_heuristic = jnp.argmax(error_diffs.at[i].set(-jnp.inf))
                a1, b1, e1, c1 = update_step_fn(alphas_op, b_op, errors_op, i, j_heuristic)

                def fallback_logic(state_and_key):
                    a_in, b_in, e_in, k_in = state_and_key
                    k_in, subkey1, subkey2 = jax.random.split(k_in, 3)

                    # === HEURISTIC 2: Search non-bound support vectors ===
                    sv_mask = (a_in > self.epsilon) & (a_in < self.C - self.epsilon)
                    sv_indices = jnp.where(sv_mask, size=n_samples, fill_value=-1)[0]
                    # The problematic line has been removed.
                    shuffled_sv_indices = jax.random.permutation(subkey1, sv_indices)
                    a2, b2, e2, c2 = search_loop_fn((a_in, b_in, e_in, i), shuffled_sv_indices)

                    # === HEURISTIC 3: Search all points (if SV search fails) ===
                    def full_scan(state):
                        a_in2, b_in2, e_in2, _ = state
                        all_indices = jnp.arange(n_samples)
                        shuffled_all_indices = jax.random.permutation(subkey2, all_indices)
                        return search_loop_fn((a_in2, b_in2, e_in2, i), shuffled_all_indices)
                    
                    # Conditionally run the full scan if the SV scan found nothing
                    a_final, b_final, e_final, c_final = jax.lax.cond(
                        c2, lambda s: (s[0], s[1], s[2], True), full_scan, (a2, b2, e2, c2)
                    )
                    return a_final, b_final, e_final, c_final, k_in

                # Chain the heuristics using lax.cond
                final_a, final_b, final_e, final_c, final_key = jax.lax.cond(
                    c1,
                    lambda _: (a1, b1, e1, c1, key_op), # <-- Corrected line
                    fallback_logic,
                    (alphas_op, b_op, errors_op, key_op)
                )

                new_count = count_op + jnp.where(final_c, 1, 0)
                return final_a, final_b, final_e, new_count, final_key

            def no_update(operands):
                return operands
            
            new_carry = jax.lax.cond(kkt_violated, perform_update, no_update, 
                                     (alphas, b, errors, alphas_changed_count, key))
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
                
            # Also ensure i != j to prevent eta from being exactly zero in some cases
            is_valid_pair = jnp.logical_and(i != j, (H - L > self.epsilon) & (eta < 0))
            return jax.lax.cond(is_valid_pair, optimization_logic, no_op, None)
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
    print("--- FINAL BENCHMARK RESULTS (With Robust Fallback Logic) ---")
    print("="*70)
    print(f"{'Implementation':>15} | {'Train Time (s)':>18} | {'Accuracy (%)':>15}")
    print("-"*70)
    print(f"{'JAX Kernel SVM':>15} | {jax_train_time:>18.4f} | {accuracy_jax*100:>15.2f}")
    print(f"{'Scikit-learn SVC':>15} | {end_time_sklearn - start_time_sklearn:>18.4f} | {accuracy_sklearn*100:>15.2f}")
    print("="*70)
