import jax
import jax.numpy as jnp
from functools import partial
import time # For benchmarking

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Imports for scikit-learn benchmark
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline


class JaxBCD_SVM:
    """
    A Linear Support Vector Machine implemented with JAX, trained using
    a Block Coordinate Descent (BCD) algorithm (specifically, a variant of SMO).

    This implementation is designed for efficiency, especially for datasets where
    the number of features is very large. It now includes an integrated
    preprocessor for standardization and normalization.

    Attributes:
        C (float): Regularization parameter.
        tol (float): Tolerance for stopping criteria (related to KKT conditions).
        max_iter (int): Maximum number of iterations over the training dataset.
        key (jax.random.PRNGKey): JAX random key for reproducibility.
        epsilon (float): A small threshold to check for meaningful changes in alpha values
                         and to identify support vectors.
        jitter (float): A small value added to the diagonal of the kernel matrix
                        to improve numerical stability during optimization.
    """
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, random_seed=42, epsilon=1e-5, jitter=1e-6):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.key = jax.random.PRNGKey(random_seed)
        self.epsilon = epsilon
        self.jitter = jitter

        # The state of the model (alphas, bias, and weights)
        self.alphas = None
        self.b = 0.0
        self.w = None
        
        # Parameters for the integrated scaler
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y):
        """
        Fits the SVM model to the training data using a JAX-idiomatic lax.scan
        for the inner loop over samples. Data is scaled and normalized internally.

        Args:
            X (jnp.ndarray): Raw training data of shape (n_samples, n_features).
            y (jnp.ndarray): Target values of shape (n_samples,). Accepts labels {0, 1} or {-1, 1}.
        """
        # --- 0. Input Validation and Label Conversion ---
        # The SVM formulation requires labels to be {-1, 1}.
        # We check if the labels are {0, 1} and convert them automatically.
        unique_labels = jnp.unique(y)
        if jnp.all(jnp.sort(unique_labels) == jnp.array([0, 1])):
            print("Labels detected in {0, 1} format. Converting to {-1, 1} for SVM training.")
            y = jnp.where(y == 0, -1, 1)
        
        # --- 1. Calculate and store scaling parameters from training data ---
        self.mean_ = jnp.mean(X, axis=0)
        self.scale_ = jnp.std(X, axis=0)
        
        # --- 2. Scale and normalize the training data ---
        # Avoid division by zero for features with no variance
        safe_scale = jnp.where(self.scale_ == 0, 1.0, self.scale_)
        X_scaled = (X - self.mean_) / safe_scale
        # L2 normalize each sample (row) for numerical stability in high dimensions
        norms = jnp.linalg.norm(X_scaled, axis=1, keepdims=True)
        # Add a small epsilon to avoid division by zero for zero-norm samples
        X_processed = X_scaled / (norms + self.epsilon)
        
        # --- 3. Sanitize data ---
        # Replace any potential NaN/inf values from division issues with zero.
        X_processed = jnp.nan_to_num(X_processed)

        # --- 4. Add a more robust diagnostic check for data variance ---
        per_feature_std = jnp.std(X_processed, axis=0)
        if jnp.all(per_feature_std < self.epsilon):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!! ERROR: Training data has zero variance after preprocessing.         !!")
            print("!! All samples are identical, so the SVM cannot be trained.            !!")
            print("!! This is likely caused by features with a very low dynamic range.    !!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return # Stop fitting

        n_samples, n_features = X_processed.shape

        # --- Initialize model state ---
        self.alphas = jnp.zeros(n_samples)
        self.b = 0.0
        self.w = jnp.zeros(n_features)
        
        # --- 5. Pre-compute indices for the anti-deadlock heuristic ---
        # This is done outside the JIT'd loop to avoid ConcretizationTypeError
        positive_indices = jnp.where(y == 1)[0]
        negative_indices = jnp.where(y == -1)[0]

        # --- JIT-compile the function for an entire epoch ---
        epoch_body_fn = self._make_epoch_body(X_processed, y, positive_indices, negative_indices)
        
        @jax.jit
        def run_epoch(carry, indices):
            # The scan function takes the body function, the initial state (carry),
            # and the array to iterate over (the shuffled indices).
            final_carry, _ = jax.lax.scan(epoch_body_fn, carry, indices)
            return final_carry

        print("Starting SVM training with JAX (using lax.scan)...")
        for iter_num in range(self.max_iter):
            # Prepare for the epoch
            self.key, subkey = jax.random.split(self.key)
            shuffled_indices = jax.random.permutation(subkey, n_samples)
            
            # The initial state ("carry") for the scan
            initial_carry = (self.alphas, self.w, self.b, self.key, 0)

            # Run one full pass over the data
            final_carry = run_epoch(initial_carry, shuffled_indices)

            # Unpack the results of the epoch
            self.alphas, self.w, self.b, self.key, alphas_changed = final_carry
            
            print(f"Iteration {iter_num + 1}/{self.max_iter} | Alpha pairs changed: {alphas_changed}")
            if alphas_changed == 0 and iter_num > 0: # Check for convergence after the first epoch
                print("Convergence reached.")
                break
                
        print("Training finished.")

    @partial(jax.jit, static_argnums=(0,))
    def decision_function(self, X):
        """
        Computes the decision function for a given set of samples.
        The decision function is the raw score, `w^T x + b`, which represents
        the signed distance to the hyperplane. Data is scaled internally before prediction.

        Args:
            X (jnp.ndarray): Samples to predict, of shape (n_test_samples, n_features).

        Returns:
            jnp.ndarray: The decision function value for each sample.
        """
        # Apply the stored scaling parameters from training
        safe_scale = jnp.where(self.scale_ == 0, 1.0, self.scale_)
        X_scaled = (X - self.mean_) / safe_scale
        # L2 normalize each sample (row) using the same logic as in fit()
        norms = jnp.linalg.norm(X_scaled, axis=1, keepdims=True)
        X_processed = X_scaled / (norms + self.epsilon)
        
        # Sanitize the processed data to handle any potential numerical issues
        X_processed = jnp.nan_to_num(X_processed)

        return X_processed @ self.w + self.b

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, X):
        """
        Predicts the class labels for a given set of samples.

        Args:
            X (jnp.ndarray): Samples to predict, of shape (n_test_samples, n_features).

        Returns:
            jnp.ndarray: Predicted labels (-1 or 1).
        """
        return jnp.sign(self.decision_function(X))

    def _make_epoch_body(self, X, y, positive_indices, negative_indices):
        """
        Creates the function that defines the logic for a single step of the
        inner loop. This function will be used inside `jax.lax.scan`.
        """
        # Create the update function once, it will be closed over by epoch_body.
        update_step_fn = self._make_update_step(X, y)
        n_samples = X.shape[0]

        def epoch_body(carry, i):
            """
            The body of the scan. Takes the current state (carry) and a sample
            index `i`, and returns the updated state.
            """
            alphas, w, b, key, alphas_changed_count = carry

            # Calculate error and check KKT conditions
            error_i = jnp.dot(w, X[i]) + b - y[i]
            kkt_violated = ((y[i] * error_i < -self.tol) & (alphas[i] < self.C)) | \
                           ((y[i] * error_i > self.tol) & (alphas[i] > 0))

            def perform_update():
                # Logic for when KKT conditions are violated: try to update alphas, w, b
                new_key, subkey = jax.random.split(key)
                
                # --- JAX-FRIENDLY HEURISTIC TO PREVENT DEADLOCK ---
                def pick_opposite_label_j():
                    # Pick a j by sampling from the pre-computed index arrays.
                    def sample_from_negatives():
                        return jax.random.choice(subkey, negative_indices)
                    def sample_from_positives():
                        return jax.random.choice(subkey, positive_indices)
                    return jax.lax.cond(y[i] == 1, sample_from_negatives, sample_from_positives)

                def pick_any_random_j():
                    # Fallback to original logic if one class is missing (degenerate case)
                    rand_j = jax.random.choice(subkey, n_samples)
                    return jnp.where(rand_j == i, (rand_j + 1) % n_samples, rand_j)
                
                # Use the heuristic only if both classes exist in the dataset
                j = jax.lax.cond(
                    (positive_indices.shape[0] > 0) & (negative_indices.shape[0] > 0),
                    pick_opposite_label_j,
                    pick_any_random_j
                )
                # --- END HEURISTIC ---

                # The update function now returns a flag indicating if a change was made
                new_alphas, new_w, new_b, changed = update_step_fn(alphas, w, b, i, j)

                # Increment the counter only if the update function reports a change
                new_count = alphas_changed_count + jnp.where(changed, 1, 0)
                
                return new_alphas, new_w, new_b, new_key, new_count

            def no_update():
                # Logic for when KKT conditions are met: state does not change
                return alphas, w, b, key, alphas_changed_count
            
            # Use lax.cond for JIT-compatibility
            new_carry_tuple = jax.lax.cond(
                kkt_violated,
                perform_update,
                no_update
            )
            
            return new_carry_tuple, None # No per-iteration output needed

        return epoch_body

    def _make_update_step(self, X, y):
        """
        Creates the JIT-compilable function for a single SMO update step.
        This is a factory function that captures X and y.
        """
        def update_step(alphas, w, b, i, j):
            """
            Performs the analytical optimization for a pair of alphas (i, j).
            This version is JAX-compliant and now returns a boolean `changed` flag.
            """
            # Store old alphas
            alpha_i_old, alpha_j_old = alphas[i], alphas[j]

            # Calculate errors for both samples
            error_i = jnp.dot(w, X[i]) + b - y[i]
            error_j = jnp.dot(w, X[j]) + b - y[j]

            # --- Compute bounds (L, H) for alpha_j using lax.cond ---
            L, H = jax.lax.cond(
                y[i] != y[j],
                lambda: (jnp.maximum(0, alpha_j_old - alpha_i_old),
                         jnp.minimum(self.C, self.C + alpha_j_old - alpha_i_old)),
                lambda: (jnp.maximum(0, alpha_i_old + alpha_j_old - self.C),
                         jnp.minimum(self.C, alpha_i_old + alpha_j_old))
            )

            # Compute eta = 2*K_ij - K_ii - K_jj
            k_ii = jnp.dot(X[i], X[i])
            k_jj = jnp.dot(X[j], X[j])
            k_ij = jnp.dot(X[i], X[j])
            # Add jitter for numerical stability
            eta = 2.0 * k_ij - k_ii - k_jj - self.jitter

            def optimization_logic():
                """The core logic for updating alphas, w, and b."""
                # Compute new alpha_j and clip it
                alpha_j_new = alpha_j_old - (y[j] * (error_i - error_j)) / eta
                alpha_j_new_clipped = jnp.clip(alpha_j_new, L, H)

                # Check if the change in alpha_j is meaningful
                delta_alpha_j = alpha_j_new_clipped - alpha_j_old
                
                # If change is negligible, do nothing and report no change
                def update_and_report_change():
                    # Compute new alpha_i based on the change in alpha_j
                    alpha_i_new = alpha_i_old - y[i] * y[j] * delta_alpha_j
                    delta_alpha_i = alpha_i_new - alpha_i_old

                    # Update alphas vector
                    new_alphas = alphas.at[i].set(alpha_i_new)
                    new_alphas = new_alphas.at[j].set(alpha_j_new_clipped)

                    # --- Update bias b ---
                    b1 = b - error_i - y[i] * delta_alpha_i * k_ii - y[j] * delta_alpha_j * k_ij
                    b2 = b - error_j - y[i] * delta_alpha_i * k_ij - y[j] * delta_alpha_j * k_jj
                    
                    new_b = jax.lax.cond(
                        (0 < alpha_i_new) & (alpha_i_new < self.C),
                        lambda: b1,
                        lambda: jax.lax.cond(
                            (0 < alpha_j_new_clipped) & (alpha_j_new_clipped < self.C),
                            lambda: b2, lambda: (b1 + b2) / 2.0
                        )
                    )
                    # --- Update weight vector w efficiently ---
                    new_w = w + delta_alpha_i * y[i] * X[i] + delta_alpha_j * y[j] * X[j]

                    return new_alphas, new_w, new_b, True

                def no_change_and_report():
                    return alphas, w, b, False

                return jax.lax.cond(
                    jnp.abs(delta_alpha_j) > self.epsilon,
                    update_and_report_change,
                    no_change_and_report
                )

            def no_op():
                """A no-op for when optimization isn't possible."""
                return alphas, w, b, False

            # This condition replaces the early 'return' statements. We only optimize
            # if there's room to move (L < H) and the objective is concave (eta < 0).
            return jax.lax.cond(
                (H - L > self.epsilon) & (eta < 0),
                optimization_logic,
                no_op
            )

        return update_step


if __name__ == '__main__':
    # --- 1. Define Benchmark Parameters ---
    feature_dimensions = [1000, 10000, 50000, 200000]
    results = []

    for n_features in feature_dimensions:
        print(f"\n{'='*20} BENCHMARKING WITH {n_features} FEATURES {'='*20}")
        # --- Generate Synthetic Data ---
        print("Generating synthetic data...")
        X, y = make_classification(
            n_samples=500,
            n_features=n_features,
            n_informative=int(n_features * 0.05),
            n_redundant=int(n_features * 0.01),
            n_classes=2,
            n_clusters_per_class=1,
            flip_y=0.05,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- 2. Train and Evaluate JAX SVM ---
        print("\n--- Training JAX BCD SVM ---")
        svm_params = {'C': 1.0, 'max_iter': 100, 'random_seed': 42, 'jitter': 1e-6}
        svm = JaxBCD_SVM(**svm_params)
        X_train_jax, y_train_jax, X_test_jax = jnp.array(X_train), jnp.array(y_train), jnp.array(X_test)

        start_time_jax = time.perf_counter()
        svm.fit(X_train_jax, y_train_jax)
        end_time_jax = time.perf_counter()
        jax_train_time = end_time_jax - start_time_jax

        y_pred_jax = svm.predict(X_test_jax)
        y_test_svm_labels = jnp.where(y_test == 0, -1, 1)
        accuracy_jax = accuracy_score(y_test_svm_labels, y_pred_jax)

        # --- 3. Train and Evaluate scikit-learn LinearSVC ---
        print("\n--- Training scikit-learn LinearSVC ---")
        sklearn_svc = Pipeline([
            ('scaler', StandardScaler()),
            ('normalizer', Normalizer()),
            ('svc', LinearSVC(
                C=svm_params['C'],
                max_iter=svm_params['max_iter'] * 20,
                tol=1e-4,
                random_state=svm_params['random_seed'],
                dual="auto"
            ))
        ])

        start_time_sklearn = time.perf_counter()
        sklearn_svc.fit(X_train, y_train)
        end_time_sklearn = time.perf_counter()
        sklearn_train_time = end_time_sklearn - start_time_sklearn

        y_pred_sklearn = sklearn_svc.predict(X_test)
        accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

        # Store results for this run
        results.append({
            'features': n_features,
            'jax_time': jax_train_time,
            'sklearn_time': sklearn_train_time,
            'jax_acc': accuracy_jax,
            'sklearn_acc': accuracy_sklearn
        })

    # --- 4. Final Benchmark Results Table ---
    print("\n\n" + "="*70)
    print("--- FINAL BENCHMARK RESULTS ---")
    print("="*70)
    print(f"{'Features':>10} | {'JAX Time (s)':>15} | {'Sklearn Time (s)':>18} | {'JAX Acc (%)':>15} | {'Sklearn Acc (%)':>18}")
    print("-"*70)

    for res in results:
        print(f"{res['features']:>10,} | {res['jax_time']:>15.4f} | {res['sklearn_time']:>18.4f} | {res['jax_acc']*100:>15.2f} | {res['sklearn_acc']*100:>18.2f}")
    
    print("="*70)


