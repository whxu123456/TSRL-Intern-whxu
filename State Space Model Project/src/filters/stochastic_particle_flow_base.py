import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.integrate import solve_bvp

tfd = tfp.distributions
tfm = tfp.math
tfs = tfp.stats


class BaseParticleFlowFilterCore:
    """
    Base code for stochastic particle flow filter.
    """
    def __init__(
        self,
        model,
        num_particles=50,
        num_steps=100,
        resample=False,
        diffusion_scale=0.03,
        jitter=1e-6,
        drift_clip=10.0
    ):
        self.model = model
        self.num_particles = num_particles
        self.N = num_particles
        self.num_steps = num_steps
        self.num_flow_steps = num_steps
        self.dlambda = 1.0 / num_steps
        self.resample = resample
        self.diffusion_scale = diffusion_scale
        self.jitter = jitter
        self.drift_clip = drift_clip

    # Universal Utility Functions
    def _sample_covariance(self, particles):
        particles = tf.convert_to_tensor(particles, dtype=tf.float32)
        mean = tf.reduce_mean(particles, axis=0)
        cov = tfs.covariance(particles, sample_axis=0, event_axis=1)
        cov += self.jitter * tf.eye(tf.shape(cov)[0], dtype=cov.dtype)
        return mean, cov

    def _normalize_log_weights(self, log_w):
        return tf.exp(tfm.log_normalize(log_w))

    def systematic_resample(self, particles, weights=None):
        if weights is None:
            weights = tf.ones([self.num_particles], dtype=tf.float32) / self.num_particles
        N = self.num_particles
        u = tf.random.uniform(shape=(), minval=0.0, maxval=1.0 / N)
        u = u + tf.cast(tf.range(N), tf.float32) / N
        cdf = tf.cumsum(weights)
        indices = tf.searchsorted(cdf, u)
        return tf.gather(particles, indices)

    def effective_sample_size(self, weights):
        return tfp.mcmc.effective_sample_size(
            weights[None, :],
            filter_threshold=0.0
        )[0]

    def _symmetrize(self, A):
        return 0.5 * (A + tf.linalg.matrix_transpose(A))

    def _regularize_matrix(self, A, eps=None):
        if eps is None:
            eps = self.jitter
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        A = self._symmetrize(A)
        d = tf.shape(A)[0]
        return A + eps * tf.eye(d, dtype=A.dtype)

    def _safe_inverse(self, A, eps=None):
        A_reg = self._regularize_matrix(A, eps)
        return tf.linalg.inv(A_reg)

    def _apply_boundaries(self, particles):
        if hasattr(self.model, "reflect_boundaries"):
            return self.model.reflect_boundaries(particles)
        return particles

    def _measurement_eval(self, x):
        if hasattr(self.model, "measurement_model"):
            return self.model.measurement_model(x)
        else:
            raise AttributeError("Model must implement measurement_model(x)")

    def _measurement_jacobian(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if hasattr(self.model, "measurement_jacobian"):
            H = self.model.measurement_jacobian(x)
            return tf.convert_to_tensor(H, dtype=tf.float32)
        # Autodiff fallback
        x_var = tf.Variable(x)
        with tf.GradientTape() as tape:
            hx = self._measurement_eval(x_var)
        H = tape.jacobian(hx, x_var)
        return tf.convert_to_tensor(H, dtype=tf.float32)

    def _get_diffusion_cov(self, dtype=tf.float32, state_dim=None):
        """
        Add state_dim parameter for dynamic dimension support
        """
        if hasattr(self.model, "Q_flow"):
            Q = tf.convert_to_tensor(self.model.Q_flow, dtype=dtype)
        elif hasattr(self.model, "Q_cov"):
            Q = tf.convert_to_tensor(self.model.Q_cov, dtype=dtype)
        elif hasattr(self.model, "prior_cov"):
            d = tf.shape(self.model.prior_cov)[0]
            Q = 0.01 * tf.eye(d, dtype=dtype)
        else:
            # Auto-create default Q, use dynamic state_dim if provided
            if state_dim is None:
                state_dim = 4  # Fallback
            Q = 0.01 * tf.eye(state_dim, dtype=dtype)

        Q = self._regularize_matrix(Q)
        return self.diffusion_scale * Q

    def _linear_beta_schedule(self):
        '''
        The linear beta schedule for comparison
        '''
        lmbda = np.linspace(0.0, 1.0, self.num_steps + 1).astype(np.float32)
        beta = lmbda.copy()
        beta_dot = np.ones_like(lmbda, dtype=np.float32)
        return beta, beta_dot

    # Optimal Homotopy Solver
    def _compute_hessians(self, z, prior_mean, prior_cov):
        """
        Compute Hessians of log-prior and log-likelihood (Gauss-Newton approx).
        """
        x = tf.convert_to_tensor(prior_mean, dtype=tf.float32)
        H = self._measurement_jacobian(x)
        R = self.model.get_measurement_cov(x)
        R_inv = self._safe_inverse(R)

        H_log_p0 = -self._safe_inverse(prior_cov)
        H_log_h = -tf.matmul(tf.matmul(H, R_inv, transpose_a=True), H)
        H_log_h = self._symmetrize(H_log_h)
        return H_log_p0, H_log_h

    def solve_optimal_beta(self, z, prior_mean, prior_cov):
        """
        Solve optimal homotopy beta(lambda) from Dai22 Theorem 3.1.
        """
        H_log_p0, H_log_h = self._compute_hessians(z, prior_mean, prior_cov)
        H_log_p0_np = H_log_p0.numpy()
        H_log_h_np = H_log_h.numpy()
        mu = float(getattr(self.model, "mu", 0.2))

        def bvp_ode(lmbda, y):
            beta = y[0]
            d2beta = np.zeros_like(beta)
            for i in range(len(beta)):
                b = beta[i]
                M = -H_log_p0_np - b * H_log_h_np
                M = 0.5 * (M + M.T)
                try:
                    M_inv = np.linalg.inv(M + self.jitter * np.eye(M.shape[0]))
                    # Nuclear norm derivative (Dai22 Remark 3.2)
                    term1 = np.trace(H_log_h_np) * np.trace(M_inv)
                    term2 = np.trace(M) * np.trace(M_inv @ M_inv @ H_log_h_np)
                    d2beta[i] = -mu * (term1 - term2)
                except np.linalg.LinAlgError:
                    d2beta[i] = 0.0
            return np.vstack((y[1], d2beta))

        def boundary_conditions(ya, yb):
            return np.array([ya[0], yb[0] - 1.0], dtype=np.float64)

        lmbda_grid = np.linspace(0.0, 1.0, self.num_steps + 1)
        y_init = np.vstack((lmbda_grid, np.ones_like(lmbda_grid)))

        try:
            sol = solve_bvp(bvp_ode, boundary_conditions, lmbda_grid, y_init, max_nodes=5000)
            if sol.success:
                beta_vals = sol.sol(lmbda_grid)[0].astype(np.float32)
                beta_dot_vals = sol.sol(lmbda_grid)[1].astype(np.float32)
                return beta_vals, beta_dot_vals
        except Exception as e:
            print(f"BVP solve failed, falling back to linear homotopy: {e}")
        return self._linear_beta_schedule()

    def predict(self, particles):
        if hasattr(self.model, "propagate_particles"):
            particles = self.model.propagate_particles(particles)
        particles = self._apply_boundaries(particles)
        return particles

