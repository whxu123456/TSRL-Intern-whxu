import tensorflow as tf
import numpy as np
from src.filters.particle_flow_base import BaseParticleFlowFilter


class SVParticleFlowFilter(BaseParticleFlowFilter):
    """
    Particle Flow Filter for Stochastic Volatility Model.
    """

    def __init__(self, model, num_particles=100, num_steps=50, method='EDH'):
        # SV model: 1D state, 1D measurement
        state_dim = 1
        meas_dim = 1
        R_cov = tf.eye(1, dtype=tf.float32)

        # Initialize base class
        super().__init__(
            num_particles=num_particles,
            state_dim=state_dim,
            meas_dim=meas_dim,
            num_flow_steps=num_steps,
            flow_method=method.split('_')[0],
            R_cov=R_cov
        )

        # SV-specific parameters
        self.model = model
        self.method = method
        self.obs_noise_mean = -1.27
        self.obs_noise_var = np.pi ** 2 / 2

        # Initialize particles
        init_std = model.sigma / np.sqrt(1 - model.alpha ** 2)
        self.particles = tf.random.normal((self.N, 1), 0, init_std, dtype=tf.float32)
        self.weights = tf.ones([self.N], dtype=tf.float32) / self.N

    def predict(self):
        """Prior prediction step for SV model."""
        noise = tf.random.normal(self.particles.shape, 0, self.model.sigma)
        self.particles = self.model.alpha * self.particles + noise
        self.flow_mags = []
        self.condition_numbers = []

    def flow_edh_exact(self, y_obs):
        """
        Pure EDH without Log-transform (Daum 2010).
        """
        R_scalar = tf.squeeze(self.R_cov)
        R_inv_scalar = tf.squeeze(self.R_inv)
        lam = 0.0

        for s in range(self.num_flow_steps):
            # 1. Compute covariance and mean (as scalars)
            P_scalar = tf.squeeze(self.compute_covariance(self.particles))
            x_bar_scalar = tf.squeeze(tf.reduce_mean(self.particles, axis=0))

            # 2. Linearize measurement model (all scalars)
            H_scalar = 0.5 * self.model.beta * tf.exp(x_bar_scalar / 2.0)
            h_bar_scalar = self.model.beta * tf.exp(x_bar_scalar / 2.0)

            # 3. Compute flow matrix A (scalar)
            HPH_scalar = H_scalar * P_scalar * H_scalar
            M_scalar = lam * HPH_scalar + R_scalar
            M_inv_scalar = 1.0 / (M_scalar + 1e-12)
            A_scalar = -0.5 * P_scalar * H_scalar * M_inv_scalar * H_scalar

            # 4. Compute flow bias b (scalar)
            e_scalar = h_bar_scalar - H_scalar * x_bar_scalar
            innov_scalar = tf.squeeze(y_obs) - e_scalar
            term_innov_scalar = P_scalar * H_scalar * R_inv_scalar * innov_scalar

            I_scalar = 1.0
            term1_scalar = (I_scalar + lam * A_scalar) * term_innov_scalar
            term2_scalar = A_scalar * x_bar_scalar
            b_scalar = (I_scalar + 2.0 * lam * A_scalar) * (term1_scalar + term2_scalar)

            # 5. Expand scalars to tensors for particle update
            A = tf.reshape(A_scalar, [1, 1])
            b = tf.reshape(b_scalar, [1, 1])

            # 6. Update particles
            drift = tf.matmul(self.particles, A, transpose_b=True) + tf.transpose(b)
            self.particles += self.dt * drift
            lam += self.dt

            # 7. Diagnostics
            J = tf.eye(1) + self.dt * A
            self._record_diagnostics(drift, J)

    def flow_ledh_exact(self, y_obs):
        """
        Pure LEDH without Log-transform (Daum 2011).
        """
        R_scalar = tf.squeeze(self.R_cov)
        R_inv_scalar = tf.squeeze(self.R_inv)
        lam = 0.0

        for s in range(self.num_flow_steps):
            # 1. Compute covariance (scalar)
            P_scalar = tf.squeeze(self.compute_covariance(self.particles))

            # 2. Per-particle linearization (all [N, 1] shaped)
            H_i = 0.5 * self.model.beta * tf.exp(self.particles / 2.0)  # [N, 1]
            h_val_i = self.model.beta * tf.exp(self.particles / 2.0)  # [N, 1]

            # 3. Per-particle flow matrix A ([N, 1])
            HPH_i = H_i * P_scalar * H_i  # [N, 1]
            M_i = lam * HPH_i + R_scalar  # [N, 1]
            M_inv_i = 1.0 / (M_i + 1e-12)  # [N, 1]
            A_i = -0.5 * P_scalar * H_i * M_inv_i * H_i  # [N, 1]

            # 4. Per-particle flow bias b ([N, 1])
            Hx_i = H_i * self.particles  # [N, 1]
            e_i = h_val_i - Hx_i  # [N, 1]
            innov_i = tf.squeeze(y_obs) - e_i  # [N, 1]
            term_innov_i = P_scalar * H_i * R_inv_scalar * innov_i  # [N, 1]

            I_scalar = 1.0
            term1_i = (I_scalar + lam * A_i) * term_innov_i  # [N, 1]
            term2_i = A_i * self.particles  # [N, 1]
            b_i = (I_scalar + 2.0 * lam * A_i) * (term1_i + term2_i)  # [N, 1]

            # 5. Update particles
            drift = A_i * self.particles + b_i  # [N, 1]
            self.particles += self.dt * drift
            lam += self.dt

            # 6. Diagnostics (J is [N, 1, 1])
            J = tf.eye(1, batch_shape=[self.N]) + self.dt * tf.expand_dims(A_i, axis=1)
            self._record_diagnostics(drift, J)

    def flow_edh_ledh_log_trans(self, y_obs):
        """
        EDH/LEDH with log(y^2) transform (Li 2017).
        """
        z_obs_trans = tf.math.log(tf.squeeze(y_obs) ** 2 + 1e-8)
        R_scalar = self.obs_noise_var
        R_inv_scalar = 1.0 / (R_scalar + 1e-12)
        H_scalar = 1.0

        lam = 0.0
        for s in range(self.num_flow_steps):
            P_scalar = tf.squeeze(self.compute_covariance(self.particles))

            HPH_scalar = H_scalar * P_scalar * H_scalar
            M_scalar = lam * HPH_scalar + R_scalar
            M_inv_scalar = 1.0 / (M_scalar + 1e-12)
            A_scalar = -0.5 * P_scalar * H_scalar * M_inv_scalar * H_scalar

            bias = tf.math.log(self.model.beta ** 2) + self.obs_noise_mean
            innovation = z_obs_trans - bias
            K_gain_scalar = P_scalar * H_scalar * R_inv_scalar

            I_scalar = 1.0
            term1_scalar = (I_scalar + lam * A_scalar) * K_gain_scalar * innovation
            b_scalar = (I_scalar + 2.0 * lam * A_scalar) * term1_scalar

            # Expand to tensors
            A = tf.reshape(A_scalar, [1, 1])
            b = tf.reshape(b_scalar, [1, 1])

            drift = tf.matmul(self.particles, A, transpose_b=True) + b
            self.particles += self.dt * drift
            lam += self.dt

            J = tf.eye(1) + self.dt * A
            self._record_diagnostics(drift, J)

    def rbf_kernel(self, X):
        """RBF Kernel matrix and gradient."""
        dists = X[:, None] - X[None, :]
        h = tf.reduce_mean(tf.abs(dists)) + 1e-6
        sq_dists = tf.square(dists)
        K = tf.exp(-sq_dists / (2 * h ** 2))
        K = tf.squeeze(K, axis=-1)
        grad_K = -(dists / (h ** 2)) * K[:, :, None]
        grad_K = tf.squeeze(grad_K, axis=-1)
        return K, grad_K

    def flow_kernel(self, y_obs):
        """Kernel PFF (Hu21 Algorithm 1)."""
        for s in range(self.num_flow_steps):
            with tf.GradientTape() as tape:
                tape.watch(self.particles)
                log_p = self.model.log_likelihood(y_obs, self.particles)

            grad_log_p = tape.gradient(log_p, self.particles)
            K, grad_K = self.rbf_kernel(self.particles)

            term1 = tf.matmul(K, grad_log_p)
            term2 = tf.reduce_sum(grad_K, axis=1, keepdims=True)
            drift = (term1 + term2) / float(self.N)
            self.particles += self.dt * drift

            self._record_diagnostics(drift, None)

    def update(self, y_obs):
        """Measurement update dispatcher."""
        if self.method == 'EDH_Log':
            self.flow_edh_ledh_log_trans(y_obs)
        elif self.method == 'EDH_Exact':
            self.flow_edh_exact(y_obs)
        elif self.method == 'LEDH_Exact':
            self.flow_ledh_exact(y_obs)
        elif self.method == 'Kernel':
            self.flow_kernel(y_obs)