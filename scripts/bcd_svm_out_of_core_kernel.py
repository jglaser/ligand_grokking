import jax
import jax.numpy as jnp
from functools import partial
import time
from tqdm.auto import tqdm

# The compute_kernel_matrix_batched helper is still needed for training
@partial(jax.jit, static_argnames=('kernel_fn', 'batch_size'))
def compute_kernel_matrix_batched(X, kernel_fn, batch_size):
    n_samples = X.shape[0]
    K = jnp.zeros((n_samples, n_samples))
    def body_fn(i, K_carry):
        start_idx = i * batch_size
        X_batch = jax.lax.dynamic_slice_in_dim(X, start_idx, batch_size, axis=0)
        gram_block = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel_fn(x1, x2))(X))(X_batch)
        row_indices_in_batch = jnp.arange(batch_size)
        mask = (start_idx + row_indices_in_batch) < n_samples
        current_K_slice = jax.lax.dynamic_slice(K_carry, (start_idx, 0), (batch_size, n_samples))
        updated_slice = jnp.where(mask[:, None], gram_block, current_K_slice)
        return jax.lax.dynamic_update_slice(K_carry, updated_slice, (start_idx, 0))
    num_batches = (n_samples + batch_size - 1) // batch_size
    K = jax.lax.fori_loop(0, num_batches, body_fn, K)
    return K

class JaxOutOfCoreKernelSVM:
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, random_seed=42, 
                 epsilon=1e-5, jitter=1e-6, predict_batch_size=128,
                 kernel='rbf', gamma=0.01, candidate_batch_size=128):
        self.C = C; self.tol = tol; self.max_iter = max_iter
        self.key = jax.random.PRNGKey(random_seed)
        self.epsilon = epsilon; self.jitter = jitter; self.predict_batch_size = predict_batch_size
        self.gamma = gamma; self.kernel_name = kernel
        self.kernel = self._get_kernel_function()
        self.alphas = None; self.b = 0.0; self.train_pairs = None; self.y_train = None
        self.X_ligand_train = None; self.X_protein_train = None
        self.candidate_batch_size = candidate_batch_size

    def _get_kernel_function(self):
        if self.kernel_name == 'rbf':
            return lambda x1, x2: jnp.exp(-self.gamma * jnp.sum((x1 - x2)**2))
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_name}")

    def fit(self, X_ligand, X_protein, train_pairs, y):
        self.X_ligand_train, self.X_protein_train = X_ligand, X_protein
        if jnp.array_equal(jnp.unique(y), jnp.array([0, 1])):
            y = jnp.where(y == 0, -1, 1)
        self.train_pairs = train_pairs; self.y_train = y
        n_samples = train_pairs.shape[0]

        print("Pre-computing Ligand kernel matrix..."); K_ligand_train = compute_kernel_matrix_batched(X_ligand, self.kernel, batch_size=32)
        print("Pre-computing Protein kernel matrix..."); K_protein_train = compute_kernel_matrix_batched(X_protein, self.kernel, batch_size=32)
        print("Kernel pre-computation finished.")

        self.alphas = jnp.zeros(n_samples); self.b = 0.0; errors = -y.astype(float)
        
        @jax.jit
        def run_epoch(carry, indices, k_lig, k_prot, pairs_data, y_data):
            epoch_body_fn = self._make_epoch_body_from_precomputed(pairs_data, y_data, k_lig, k_prot)
            final_carry, _ = jax.lax.scan(epoch_body_fn, carry, indices)
            return final_carry

        print("Starting SVM training with optimized memory access...")
        for iter_num in range(self.max_iter):
            self.key, subkey = jax.random.split(self.key); shuffled_indices = jax.random.permutation(subkey, n_samples)
            initial_carry = (self.alphas, self.b, errors, 0, subkey)
            final_carry = run_epoch(initial_carry, shuffled_indices, K_ligand_train, K_protein_train, train_pairs, y)
            self.alphas, self.b, errors, alphas_changed, _ = final_carry
            print(f"Iteration {iter_num + 1}/{self.max_iter} | Alpha pairs changed: {alphas_changed}")
            if alphas_changed == 0 and iter_num > 1: print("Convergence reached."); break
        print("Training finished.")

    def _parallel_search_for_j(self, alphas, b, errors, i, key, update_step_fn):
        n_samples = alphas.shape[0]
        key, subkey = jax.random.split(key)
        candidate_js = jax.random.randint(subkey, (self.candidate_batch_size,), 0, n_samples)

        vmapped_update = jax.vmap(update_step_fn, in_axes=(None, None, None, None, 0))
        
        new_alphas_batch, new_bs_batch, new_errors_batch, changed_batch = vmapped_update(
            alphas, b, errors, i, candidate_js
        )

        def calculate_improvement(new_alphas, changed):
            improvement = jnp.sum(jnp.abs(new_alphas - alphas)) 
            return jnp.where(changed, improvement, -1.0) 

        improvements = jax.vmap(calculate_improvement)(new_alphas_batch, changed_batch)
        best_candidate_idx = jnp.argmax(improvements)

        best_alphas = new_alphas_batch[best_candidate_idx]
        best_b = new_bs_batch[best_candidate_idx]
        best_errors = new_errors_batch[best_candidate_idx]
        was_changed = changed_batch[best_candidate_idx]

        return best_alphas, best_b, best_errors, was_changed, key

    def _make_epoch_body_from_precomputed(self, pairs, y, K_lig, K_prot):
        n_samples = pairs.shape[0]
        update_step_fn = self._make_update_step_from_precomputed(pairs, y, K_lig, K_prot)
        
        def epoch_body(carry, i):
            alphas, b, errors, alphas_changed_count, key = carry
            error_i = errors[i]
            kkt_violates = ((y[i] * error_i < -self.tol) & (alphas[i] < self.C)) | ((y[i] * error_i > self.tol) & (alphas[i] > 0))
            
            def perform_update(operands):
                a_op, b_op, e_op, count_op, key_op = operands
                j_heuristic = jnp.argmax(jnp.abs(error_i - e_op).at[i].set(-jnp.inf))
                a1, b1, e1, c1 = update_step_fn(a_op, b_op, e_op, i, j_heuristic)
                
                def fallback_logic(state_and_key):
                    a_in, b_in, e_in, k_in = state_and_key
                    # The sequential search is replaced by the parallel one
                    a_final, b_final, e_final, c_final, k_out = self._parallel_search_for_j(
                        a_in, b_in, e_in, i, k_in, update_step_fn
                    )
                    return a_final, b_final, e_final, c_final, k_out
                
                final_a, final_b, final_e, final_c, final_key = jax.lax.cond(
                    c1, 
                    lambda _: (a1, b1, e1, c1, key_op), 
                    fallback_logic, 
                    (a_op, b_op, e_op, key_op)
                )

                new_count = count_op + jnp.where(final_c, 1, 0)
                return final_a, final_b, final_e, new_count, final_key
            
            return jax.lax.cond(kkt_violates, perform_update, lambda op: op, carry), None
        
        return epoch_body

    def _make_update_step_from_precomputed(self, pairs, y, K_lig, K_prot):
        def update_step(alphas, b, errors, i, j):
            lig_i, prot_i = pairs[i]; lig_j, prot_j = pairs[j]
            k_ii = K_lig[lig_i, lig_i] * K_prot[prot_i, prot_i]
            k_jj = K_lig[lig_j, lig_j] * K_prot[prot_j, prot_j]
            k_ij = K_lig[lig_i, lig_j] * K_prot[prot_i, prot_j]
            alpha_i_old, alpha_j_old = alphas[i], alphas[j]
            error_i, error_j = errors[i], errors[j]
            L, H = jax.lax.cond(y[i] != y[j],
                lambda: (jnp.maximum(0, alpha_j_old - alpha_i_old), jnp.minimum(self.C, self.C + alpha_j_old - alpha_i_old)),
                lambda: (jnp.maximum(0, alpha_i_old + alpha_j_old - self.C), jnp.minimum(self.C, alpha_i_old + alpha_j_old)))
            eta = 2.0 * k_ij - k_ii - k_jj - self.jitter
            def optimization_logic(_):
                alpha_j_new = alpha_j_old - (y[j] * (error_i - error_j)) / eta
                alpha_j_new_clipped = jnp.clip(alpha_j_new, L, H)
                delta_alpha_j = alpha_j_new_clipped - alpha_j_old
                def update_and_report(_):
                    alpha_i_new = alpha_i_old - y[i] * y[j] * delta_alpha_j
                    delta_alpha_i = alpha_i_new - alpha_i_old
                    new_alphas = alphas.at[i].set(alpha_i_new).at[j].set(alpha_j_new_clipped)
                    b1=b-error_i-y[i]*delta_alpha_i*k_ii-y[j]*delta_alpha_j*k_ij
                    b2=b-error_j-y[i]*delta_alpha_i*k_ij-y[j]*delta_alpha_j*k_jj
                    cond1=jnp.logical_and(0<alpha_i_new,alpha_i_new<self.C)
                    cond2=jnp.logical_and(0<alpha_j_new_clipped,alpha_j_new_clipped<self.C)
                    new_b=jax.lax.cond(cond1,lambda:b1,lambda:jax.lax.cond(cond2,lambda:b2,lambda:(b1+b2)/2.0))
                    k_lig_row_i = K_lig[lig_i, :]; k_prot_row_i = K_prot[prot_i, :]
                    k_lig_row_j = K_lig[lig_j, :]; k_prot_row_j = K_prot[prot_j, :]
                    all_lig_indices, all_prot_indices = pairs[:, 0], pairs[:, 1]
                    k_i_all = k_lig_row_i[all_lig_indices] * k_prot_row_i[all_prot_indices]
                    k_j_all = k_lig_row_j[all_lig_indices] * k_prot_row_j[all_prot_indices]
                    new_errors = errors + delta_alpha_i*y[i]*k_i_all + delta_alpha_j*y[j]*k_j_all + (new_b - b)
                    return new_alphas, new_b, new_errors, True
                return jax.lax.cond(jnp.abs(delta_alpha_j) > self.epsilon, update_and_report, lambda _: (alphas, b, errors, False), None)
            is_valid_pair = jnp.logical_and(i != j, (H - L > self.epsilon) & (eta < 0))
            return jax.lax.cond(is_valid_pair, optimization_logic, lambda _: (alphas, b, errors, False), None)
        return update_step

    def decision_function(self, test_ligand_features, test_protein_features, test_pairs, sv_batch_size=32):
        sv_indices = jnp.where(self.alphas > self.epsilon)[0]
        sv_alphas_y = self.alphas[sv_indices] * self.y_train[sv_indices]
        sv_pairs = self.train_pairs[sv_indices]
        num_svs = len(sv_indices)

        print(f"Making predictions for {len(test_pairs)} pairs using {num_svs} SVs.")

        @partial(jax.jit, static_argnames=('self',))
        def predict_batch_kernel(self, batch_test_lig_feats, batch_test_prot_feats,
                                 sv_lig_feats, sv_prot_feats, sv_alphas_y_vals):

            def score_one_test_pair(test_lig_feat, test_prot_feat):
                x_test = jnp.concatenate([test_lig_feat, test_prot_feat])

                def get_kernel_vs_one_sv(sv_lig_feat, sv_prot_feat):
                    x_sv = jnp.concatenate([sv_lig_feat, sv_prot_feat])
                    return self.kernel(x_test, x_sv)

                kernel_values = jax.vmap(get_kernel_vs_one_sv)(sv_lig_feats, sv_prot_feats)
                return jnp.dot(kernel_values, sv_alphas_y_vals)

            return jax.vmap(score_one_test_pair)(batch_test_lig_feats, batch_test_prot_feats)

        all_scores = jnp.zeros(len(test_pairs))

        # Outer loop: Iterate over test pairs in batches
        for i in tqdm(range(0, len(test_pairs), self.predict_batch_size), desc="Predicting in batches"):
            batch_test_pairs = test_pairs[i:i+self.predict_batch_size]
            batch_test_ligand_indices = batch_test_pairs[:, 0]
            batch_test_protein_indices = batch_test_pairs[:, 1]
            batch_test_ligand_features = test_ligand_features[batch_test_ligand_indices]
            batch_test_protein_features = test_protein_features[batch_test_protein_indices]

            batch_scores = jnp.zeros(len(batch_test_pairs))

            # Inner loop: Iterate over support vectors in batches
            for j in range(0, num_svs, sv_batch_size):
                sv_batch_indices = sv_indices[j:j+sv_batch_size]
                sv_batch_alphas_y = sv_alphas_y[j:j+sv_batch_size]
                sv_batch_pairs = sv_pairs[j:j+sv_batch_size]

                sv_batch_ligand_indices = sv_batch_pairs[:, 0]
                sv_batch_protein_indices = sv_batch_pairs[:, 1]

                sv_batch_ligand_features = self.X_ligand_train[sv_batch_ligand_indices]
                sv_batch_protein_features = self.X_protein_train[sv_batch_protein_indices]

                # Calculate partial scores against the current batch of SVs
                partial_scores = predict_batch_kernel(
                    self, batch_test_ligand_features, batch_test_protein_features,
                    sv_batch_ligand_features, sv_batch_protein_features,
                    sv_batch_alphas_y
                )
                batch_scores += partial_scores

            all_scores = all_scores.at[i:i+len(batch_scores)].set(batch_scores + self.b)

        return all_scores
