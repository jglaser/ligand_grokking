import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, tree_util
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from functools import partial
import optax
from tqdm.auto import trange
import numpy as np

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optional sklearn for metrics
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SparseVariationalGPClassifier:
    """
    A scalable Deep Kernel Learning model for BINARY CLASSIFICATION,
    adapted from the original regression implementation.
    """

    def __init__(self, n_inducing_points: int = 100, kernel='rbf', kernel_kwargs=None,
                 kernel_hparams_init=None, learning_rate: float = 1e-3, jitter: float = 1e-4,
                 hparam_prior=(2.0, 2.0), train_inducing_points: bool = False, ard: bool = False,
                 fit_mean: bool = True, log_callback=None, sampler: str = 'adam',
                 temperature: float = 1.0, cv_fn=None, encoder=None, encoder_params=None,
                 mc_samples: int = 20):
        self.n_inducing_points = n_inducing_points
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
        self.kernel_hparams_init = kernel_hparams_init
        self.learning_rate = learning_rate
        self.jitter = jitter
        self.hparam_prior = hparam_prior
        self.train_inducing_points = train_inducing_points
        self.ard = ard
        self.fit_mean = fit_mean
        self.log_callback = log_callback
        self.sampler = sampler
        self.temperature = temperature
        self.cv_fn = cv_fn
        self.encoder = encoder
        self.encoder_params = encoder_params
        self.mc_samples = mc_samples # Number of samples for expected log likelihood

        self.params = self.opt_state = self.optimizer = self.bias_state = None

    @property
    def kernel_fn(self):
        if self.kernel == 'rbf': return self._default_rbf_kernel
        elif callable(self.kernel): return self.kernel
        else: raise ValueError("kernel must be a callable function or the string 'rbf'")

    def __getstate__(self):
        # ... (unchanged from original)
        if self.params is None: raise RuntimeError("Model has not been trained.")
        state = self.__dict__.copy()
        state['params'] = jax.tree.map(np.asarray, jax.device_get(self.params))
        if self.bias_state is not None: state['bias_state'] = jax.tree.map(np.asarray, jax.device_get(self.bias_state))
        if 'kernel_kwargs' in state and state['kernel_kwargs'] is not None:
            state['kernel_kwargs'] = {k: v for k, v in state['kernel_kwargs'].items() if not hasattr(v, 'devices')}
        for key in ['opt_state', 'optimizer', 'log_callback', 'cv_fn']:
            if key in state: del state[key]
        return state

    def __setstate__(self, state):
        # ... (unchanged from original)
        self.__dict__.update(state)
        self.params = jax.tree.map(jnp.asarray, self.params)
        if 'bias_state' in state and self.bias_state is not None:
            self.bias_state = jax.tree.map(jnp.asarray, self.bias_state)
        self.optimizer = self.opt_state = None

    @staticmethod
    def _default_rbf_kernel(X1, X2, log_gamma, diag=False, **kwargs):
        # ... (unchanged from original)
        gamma = jnp.exp(log_gamma)
        if diag: return jnp.ones(X1.shape[0])
        if X2 is None: X2 = X1
        dist_sq = (X1[:, None, :] - X2[None, :, :]) ** 2
        scaled_dist_sq = jnp.sum(gamma * dist_sq, axis=-1)
        return jnp.exp(-scaled_dist_sq)

    def _compute_kernel_init(self, X1, hparams_log, encoder_params=None, X2=None):
        # ... (unchanged from original)
        if self.encoder:
            X1 = self.encoder(encoder_params, X1)
            if X2 is not None: X2 = self.encoder(encoder_params, X2)
        return self.kernel_fn(X1, X2, **hparams_log, **self.kernel_kwargs)

    @staticmethod
    @partial(jit, static_argnums=(4, 5, 6, 7, 8, 9, 10))
    def _negative_elbo(params, X_batch, y_batch, key, n_samples,
                       encoder_fn, kernel_fn, kernel_kwargs_tuple, jitter,
                       hparam_prior, mc_samples):
        """
        MODIFIED: Calculates the negative ELBO for binary classification.
        The Gaussian likelihood is replaced with an expectation of the Bernoulli likelihood.
        """
        kernel_kwargs = dict(kernel_kwargs_tuple)
        Z, m, L_S = params['Z'], params['m'], params['L_S']
        kernel_hparams_log = params['kernel_hparams']
        encoder_params, mean_offset = params.get('encoder_params'), params.get('mean_offset', 0.0)
        M, B = Z.shape[0], X_batch.shape[0]

        X_batch_encoded, Z_encoded = (encoder_fn(encoder_params, X_batch), encoder_fn(encoder_params, Z)) if encoder_fn else (X_batch, Z)

        # --- Variational Distribution q(f) ---
        K_mm = kernel_fn(Z_encoded, None, **kernel_hparams_log, **kernel_kwargs)
        K_bm = kernel_fn(X_batch_encoded, Z_encoded, **kernel_hparams_log, **kernel_kwargs)
        K_bb_diag = kernel_fn(X_batch_encoded, None, diag=True, **kernel_hparams_log, **kernel_kwargs)
        L_K = jnp.linalg.cholesky(K_mm + jnp.eye(M) * jitter)
        S = L_S @ L_S.T

        alpha_ = jax.scipy.linalg.cho_solve((L_K, True), m)
        mu_pred = K_bm @ alpha_
        K_mm_inv_K_mb = jax.scipy.linalg.cho_solve((L_K, True), K_bm.T)
        q_bb_diag = jnp.sum(K_bm * K_mm_inv_K_mb.T, axis=1)
        trace_term_var = jnp.sum((K_mm_inv_K_mb.T @ S) * K_mm_inv_K_mb.T, axis=1)
        sigma_sq = K_bb_diag - q_bb_diag + trace_term_var

        # --- Expected Log Likelihood (Bernoulli) using Monte Carlo ---
        # 1. Sample from the latent posterior q(f)
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(subkey, shape=(mc_samples, B))
        f_samples = mu_pred + jnp.sqrt(jnp.maximum(sigma_sq, 1e-12)) * epsilon

        # 2. Compute Bernoulli log likelihood for each sample
        # Note: jax.nn.log_sigmoid(f) is more stable than log(sigmoid(f))
        log_likelihood_samples = y_batch * jax.nn.log_sigmoid(f_samples) + \
                               (1 - y_batch) * jax.nn.log_sigmoid(-f_samples)

        # 3. Average over samples to get expectation
        expected_log_likelihood = jnp.mean(log_likelihood_samples, axis=0)
        scaled_log_likelihood = (n_samples / B) * jnp.sum(expected_log_likelihood)

        # --- KL Divergence (unchanged) ---
        K_mm_inv_S = jax.scipy.linalg.cho_solve((L_K, True), S)
        trace_term_kl = jnp.trace(K_mm_inv_S)
        log_det_K = 2 * jnp.sum(jnp.log(jnp.diag(L_K) + 1e-12))
        log_det_S = 2 * jnp.sum(jnp.log(jnp.diag(L_S) + 1e-12))
        KL = 0.5 * (trace_term_kl + m.T @ alpha_ - M + log_det_K - log_det_S)

        # --- Priors (unchanged, but alpha_prior is removed) ---
        total_prior_loss = 0.0
        if hparam_prior:
            shape, rate = hparam_prior
            hparams_real = jax.tree.map(jnp.exp, kernel_hparams_log)
            hparam_losses = jax.tree.map(lambda h, log_h: rate * h - (shape - 1) * log_h, hparams_real, kernel_hparams_log)
            total_prior_loss += jnp.sum(jnp.array([jnp.sum(leaf) for leaf in tree_util.tree_leaves(hparam_losses)]))

        # --- Final ELBO ---
        physical_potential = -(scaled_log_likelihood - KL) + total_prior_loss
        gammas = jnp.exp(kernel_hparams_log['log_gamma'])
        pr_val = (jnp.sum(gammas)**2) / (jnp.sum(gammas**2) + 1e-8)

        aux_metrics = {'physical_potential': physical_potential, 'kl_divergence': KL, 'participation_ratio': pr_val}
        return physical_potential, aux_metrics

    def _create_optimizer(self, key=None):
        # ... (unchanged from original)
        if self.sampler == 'adam': return optax.adam(self.learning_rate)
        elif self.sampler == 'sgld':
             if key is None: raise ValueError("SGLD sampler requires a JAX random key.")
             def add_noise(learning_rate, temperature, noise_key):
                def init_fn(params): return {'key': noise_key}
                def update_fn(updates, state, params=None):
                    updates_flat, treedef = tree_util.tree_flatten(updates)
                    keys = jax.random.split(state['key'], len(updates_flat) + 1)
                    key, subkeys = keys[0], keys[1:]
                    noise = [jnp.sqrt(2 * learning_rate * temperature) * jax.random.normal(k, p.shape)
                             for k, p in zip(subkeys, updates_flat)]
                    noisy_updates_flat = [(u + n) for u, n in zip(updates_flat, noise)]
                    return tree_util.tree_unflatten(treedef, noisy_updates_flat), {'key': key}
                return optax.GradientTransformation(init_fn, update_fn)
             return optax.chain(optax.sgd(self.learning_rate), add_noise(self.learning_rate, self.temperature, key))
        else: raise ValueError(f"Unknown sampler: {self.sampler}")


    def fit(self, X_train, y_train, epochs=100, batch_size=128, random_seed=0, wandb_run=None, bias_potential=None, mesh=None):
        """
        MODIFIED: `log_alpha` initialization removed as it's not needed for classification.
        """
        key = jax.random.PRNGKey(random_seed)
        n_samples, n_features = X_train.shape
        n_batches = n_samples // batch_size

        if self.params is None:
            key, subkey = jax.random.split(key)
            Z_init_indices = jax.random.choice(subkey, n_samples, (self.n_inducing_points,), replace=False)
            Z = X_train[Z_init_indices]
            m = jnp.zeros(self.n_inducing_points)
            ard_dim = self.encoder.output_dim if self.encoder else n_features
            log_gamma = jnp.zeros(ard_dim) if self.ard else jnp.array(0.0)
            kernel_hparams_log = self.kernel_hparams_init or {'log_gamma': log_gamma}
            # The `log_alpha` for regression noise variance is removed
            self.params = {'Z': Z, 'm': m, 'L_S': None, 'kernel_hparams': kernel_hparams_log}
            if self.encoder: self.params['encoder_params'] = self.encoder_params
            K_mm_init = self._compute_kernel_init(Z, kernel_hparams_log, self.params.get('encoder_params'))
            self.params['L_S'] = jnp.linalg.cholesky(K_mm_init + jnp.eye(self.n_inducing_points) * self.jitter)
            if self.fit_mean: self.params['mean_offset'] = 0.0 # Start mean offset at 0 for classification

            key, opt_key = jax.random.split(key)
            self.optimizer = self._create_optimizer(key=opt_key)
            self.opt_state = self.optimizer.init(self._get_trainable_params())
            if bias_potential and hasattr(bias_potential, 'init_state'):
                self.bias_state = bias_potential.init_state(self.params, n_samples)

            if mesh:
                replicated_sharding = NamedSharding(mesh, PartitionSpec())
                self.params = jax.device_put(self.params, replicated_sharding)
                self.opt_state = jax.device_put(self.opt_state, replicated_sharding)
                if self.bias_state: self.bias_state = jax.device_put(self.bias_state, replicated_sharding)

            self.X_batches = X_train[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
            self.y_batches = y_train[:n_batches * batch_size].reshape(n_batches, batch_size)
            if mesh:
                batch_sharding = NamedSharding(mesh, PartitionSpec(None, 'data'))
                self.X_batches = jax.device_put(self.X_batches, batch_sharding)
                self.y_batches = jax.device_put(self.y_batches, batch_sharding)

        kernel_kwargs_tuple = tuple(sorted(self.kernel_kwargs.items()))

        # JIT compile the epoch step function
        epoch_step_jit = self._jit_epoch_step

        pbar = trange(epochs)
        for epoch in pbar:
            key, epoch_key = jax.random.split(key)

            self.params, self.opt_state, self.bias_state, avg_metrics = epoch_step_jit(
                self.params, self.opt_state, self.bias_state, self.X_batches, self.y_batches,
                epoch_key, n_samples, self.train_inducing_points, self.fit_mean,
                bias_potential, self.cv_fn, self.optimizer, self.encoder,
                self.kernel_fn, kernel_kwargs_tuple, self.jitter,
                self.hparam_prior, self.mc_samples
            )

            pbar.set_description(f"Epoch {epoch+1}/{epochs} | Loss: {avg_metrics['loss']:.4f}")

            if wandb_run and WANDB_AVAILABLE:
                log_data = {"epoch": epoch, **avg_metrics}
                if self.log_callback:
                    custom_metrics = self.log_callback(self, epoch, avg_metrics, self.params, self.bias_state)
                    if custom_metrics: log_data.update(custom_metrics)
                wandb_run.log(log_data)
        return self

    @staticmethod
    @partial(jit, static_argnames=('train_inducing_points', 'fit_mean', 'bias_potential_fn',
                                  'cv_fn', 'optimizer', 'encoder_fn', 'kernel_fn',
                                  'n_samples', 'kernel_kwargs_tuple',
                                  'jitter', 'hparam_prior', 'mc_samples'))
    def _jit_epoch_step(params, opt_state, bias_state, X_batches, y_batches, key, n_samples,
                        train_inducing_points, fit_mean, bias_potential_fn, cv_fn, optimizer,
                        encoder_fn, kernel_fn, kernel_kwargs_tuple, jitter, hparam_prior, mc_samples):
        """A pure, JIT-compatible function for a full epoch of training."""
        n_batches = X_batches.shape[0]
        batch_indices = jax.random.permutation(key, n_batches)

        def body_fun(i, carry):
            params, opt_state, bias_state, epoch_metrics, key = carry
            batch_idx = batch_indices[i]
            X_batch, y_batch = X_batches[batch_idx], y_batches[batch_idx]

            # Perform a local, on-device shuffle of the current mini-batch shard
            key, shuffle_key, elbo_key = jax.random.split(key, 3)
            p = jax.random.permutation(shuffle_key, X_batch.shape[0])
            X_batch, y_batch = X_batch[p], y_batch[p]

            def loss_fn_with_bias(p):
                physical_potential, aux = SparseVariationalGPClassifier._negative_elbo(
                    p, X_batch, y_batch, elbo_key, n_samples, encoder_fn, kernel_fn, kernel_kwargs_tuple,
                    jitter, hparam_prior, mc_samples)
                cv_val = cv_fn(p, aux) if cv_fn else 0.0
                bias_energy = 0.0
                new_bias_state = bias_state
                if bias_potential_fn is not None:
                    bias_energy, new_bias_state = bias_potential_fn(cv_val, bias_state)
                total_loss = physical_potential + bias_energy
                aux.update({'loss': total_loss, 'bias_energy': bias_energy, 'cv_val': cv_val})
                return total_loss, (new_bias_state, aux)

            (loss, (new_bias_state, aux_metrics)), grads = value_and_grad(loss_fn_with_bias, has_aux=True)(params)

            trainable_grads = {k: grads.get(k) for k in params if k not in ['Z'] or train_inducing_points}
            if not fit_mean: trainable_grads.pop('mean_offset', None)
            trainable_params = {k: params[k] for k in trainable_grads.keys()}

            updates, new_opt_state = optimizer.update(trainable_grads, opt_state, trainable_params)
            updated_trainable_params = optax.apply_updates(trainable_params, updates)
            new_params = params.copy()
            new_params.update(updated_trainable_params)

            new_epoch_metrics = jax.tree.map(lambda a, b: a + b, epoch_metrics, aux_metrics)

            return new_params, new_opt_state, new_bias_state, new_epoch_metrics, key

        initial_metrics = {'loss': 0.0, 'physical_potential': 0.0, 'kl_divergence': 0.0, 'participation_ratio': 0.0, 'cv_val': 0.0, 'bias_energy': 0.0}
        final_params, final_opt_state, final_bias_state, total_metrics, _ = jax.lax.fori_loop(
            0, n_batches, body_fun, (params, opt_state, bias_state, initial_metrics, key)
        )

        avg_metrics = jax.tree.map(lambda x: x / n_batches, total_metrics)
        return final_params, final_opt_state, final_bias_state, avg_metrics

    def _get_trainable_params(self):
        # ... (unchanged from original)
        trainable_params = self.params.copy()
        if not self.train_inducing_points: del trainable_params['Z']
        if not self.fit_mean and 'mean_offset' in trainable_params: del trainable_params['mean_offset']
        return trainable_params

    @staticmethod
    @partial(jit, static_argnames=('encoder_fn', 'kernel_fn', 'n_inducing_points', 'jitter', 'kernel_kwargs_tuple'))
    def _predict_latent_jit(params, X_new, encoder_fn, kernel_fn, kernel_kwargs_tuple, n_inducing_points, jitter):
        # This is the original _predict_jit, renamed to reflect it predicts the latent function
        kernel_kwargs = dict(kernel_kwargs_tuple)
        Z, m, L_S = params['Z'], params['m'], params['L_S']
        mean_offset, kernel_hparams_log = params.get('mean_offset', 0.0), params['kernel_hparams']
        encoder_params = params.get('encoder_params')

        X_new_encoded, Z_encoded = (encoder_fn(encoder_params, X_new), encoder_fn(encoder_params, Z)) if encoder_fn else (X_new, Z)

        K_mm = kernel_fn(Z_encoded, None, **kernel_hparams_log, **kernel_kwargs)
        K_nm = kernel_fn(X_new_encoded, Z_encoded, **kernel_hparams_log, **kernel_kwargs)
        K_nn_diag = kernel_fn(X_new_encoded, None, diag=True, **kernel_hparams_log, **kernel_kwargs)
        L_K = jnp.linalg.cholesky(K_mm + jnp.eye(n_inducing_points) * jitter)
        S = L_S @ L_S.T

        alpha_ = jax.scipy.linalg.cho_solve((L_K, True), m)
        pred_mean = (K_nm @ alpha_) + mean_offset

        K_mm_inv_K_mn = jax.scipy.linalg.cho_solve((L_K, True), K_nm.T)
        q_nn_diag = jnp.sum(K_nm * K_mm_inv_K_mn.T, axis=1)
        trace_term = jnp.sum((K_mm_inv_K_mn.T @ S) * K_mm_inv_K_mn.T, axis=1)
        pred_var = K_nn_diag - q_nn_diag + trace_term
        return pred_mean, pred_var

    def predict_proba(self, X_new, params=None, mesh=None):
        """
        NEW: Predicts the probability of class 1.
        """
        if params is None: params = self.params
        if mesh:
            sharding = NamedSharding(mesh, PartitionSpec('data'))
            X_new = jax.device_put(X_new, sharding)

        kernel_kwargs_tuple = tuple(sorted(self.kernel_kwargs.items()))

        pred_mean, pred_var = self._predict_latent_jit(
            params, X_new, self.encoder, self.kernel_fn,
            kernel_kwargs_tuple, self.n_inducing_points, self.jitter
        )

        # Approximate the predictive probability
        # p(y=1|x) = E[sigmoid(f(x))] ≈ sigmoid(μ / sqrt(1 + π/8 * σ²))
        kappa = 1. / jnp.sqrt(1. + jnp.pi * pred_var / 8.)
        return jax.nn.sigmoid(kappa * pred_mean)

    def predict(self, X_new, params=None, mesh=None):
        """
        NEW: Predicts the class label (0 or 1).
        """
        probabilities = self.predict_proba(X_new, params=params, mesh=mesh)
        return (probabilities > 0.5).astype(jnp.int32)

    def score(self, X_test, y_test, params=None, mesh=None):
        """
        MODIFIED: Calculates accuracy instead of R^2.
        """
        y_pred = self.predict(X_test, params=params, mesh=mesh)
        return jnp.mean(y_test == y_pred)

    def score_auc(self, X_test, y_test, params=None, mesh=None):
        """
        NEW: Calculates AUC score. Requires sklearn.
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required to compute AUC score. Please install it.")
        y_prob = self.predict_proba(X_test, params=params, mesh=mesh)
        return roc_auc_score(y_test, y_prob)


# --- Example Usage for Classification---
if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    N, D = 500, 2

    # Create a simple synthetic classification dataset
    X = jax.random.uniform(key, shape=(N, D)) * 4 - 2
    # A non-linear decision boundary
    y_latent = jnp.sin(X[:, 0] * 2) + X[:, 1]
    y = (y_latent > jnp.median(y_latent)).astype(jnp.int32)

    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    print("--- Training Sparse GP Classifier (Adam) ---")
    svgpc = SparseVariationalGPClassifier(
        n_inducing_points=50,
        learning_rate=0.01,
        train_inducing_points=False,
        ard=True
    )
    svgpc.fit(X_train, y_train, epochs=300, batch_size=128)

    accuracy = svgpc.score(X_test, y_test)
    print(f"\nFinal Accuracy: {accuracy:.4f}")

    if SKLEARN_AVAILABLE:
        auc = svgpc.score_auc(X_test, y_test)
        print(f"Final AUC: {auc:.4f}")

