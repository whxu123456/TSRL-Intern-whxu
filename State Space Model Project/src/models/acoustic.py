import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class AcousticConfig:
    """
    Acoustic tracking model used for Li17 experiments.
    """

    def __init__(self):
        self.T = 40
        self.dt = 1.0

        self.n_targets = 4
        self.state_dim_per_target = 4   # [x, y, vx, vy]
        self.state_dim = self.n_targets * self.state_dim_per_target
        self.meas_dim = 25
        self.n_particles = 500

        # sensor layout: 5x5 grid in [0,40]x[0,40]
        self.sensors_pos = tf.constant([
            [0, 0], [10, 0], [20, 0], [30, 0], [40, 0],
            [0, 10], [10, 10], [20, 10], [30, 10], [40, 10],
            [0, 20], [10, 20], [20, 20], [30, 20], [40, 20],
            [0, 30], [10, 30], [20, 30], [30, 30], [40, 30],
            [0, 40], [10, 40], [20, 40], [30, 40], [40, 40]
        ], dtype=tf.float32)

        # measurement model parameters
        # z_j = sum_c Amp / (||r_c - s_j|| + d0) + noise
        self.Amp = 10.0
        self.d0 = 0.1
        self.sigma_w2 = 0.01
        self.sigma_w = np.sqrt(self.sigma_w2).astype(np.float32)

        self.R_cov = tf.eye(self.meas_dim, dtype=tf.float32) * self.sigma_w2
        self.R_inv = tf.linalg.inv(self.R_cov)
        self.R_chol = tf.linalg.cholesky(self.R_cov)

        # truth initial condition
        self.x0_truth = tf.constant([
            12.0,  6.0,   0.001,  0.001,
            32.0, 32.0,  -0.001, -0.005,
            20.0, 13.0,  -0.1,    0.01,
            15.0, 35.0,   0.002,  0.002
        ], dtype=tf.float32)

        self.x0_mean = tf.identity(self.x0_truth)

        # initial covariance for filtering particles
        # position std = 10, velocity std = 1
        p0_block = np.diag([100.0, 100.0, 1.0, 1.0]).astype(np.float32)
        self.P0 = tf.constant(np.kron(np.eye(self.n_targets), p0_block), dtype=tf.float32)
        self.P0_chol = tf.linalg.cholesky(self.P0)

        self.x0_cov = tf.identity(self.P0)

        # truth process covariance (for generating x_true)
        q_true_sub = (1.0 / 20.0) * np.array([
            [1.0 / 3.0, 0.0, 0.5, 0.0],
            [0.0, 1.0 / 3.0, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0]
        ], dtype=np.float32)
        self.Q_true = tf.constant(np.kron(np.eye(self.n_targets), q_true_sub), dtype=tf.float32)
        self.Q_true_chol = tf.linalg.cholesky(self.Q_true)

        # filter process covariance (for particle propagation)
        q_filter_sub = np.array([
            [3.0, 0.0, 0.1, 0.0],
            [0.0, 3.0, 0.0, 0.1],
            [0.1, 0.0, 0.03, 0.0],
            [0.0, 0.1, 0.0, 0.03]
        ], dtype=np.float32)
        self.Q_cov = tf.constant(np.kron(np.eye(self.n_targets), q_filter_sub), dtype=tf.float32)
        self.Q_chol = tf.linalg.cholesky(self.Q_cov)

        # keep a generic process covariance alias for compatibility
        self.Q = tf.identity(self.Q_cov)

        # constant velocity transition matrix
        f_sub = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.F = tf.constant(np.kron(np.eye(self.n_targets), f_sub), dtype=tf.float32)

        # spatial boundaries
        self.min_pos = 0.0
        self.max_pos = 40.0


class AcousticModel:
    def __init__(self, config: AcousticConfig):
        self.cfg = config
        self._prior_dist = tfd.MultivariateNormalTriL(
            loc=self.cfg.x0_mean,
            scale_tril=self.cfg.P0_chol
        )
        self._true_process_noise_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.cfg.state_dim, dtype=tf.float32),
            scale_tril=self.cfg.Q_true_chol
        )
        self._filter_process_noise_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.cfg.state_dim, dtype=tf.float32),
            scale_tril=self.cfg.Q_chol
        )
        self._meas_noise_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.cfg.meas_dim, dtype=tf.float32),
            scale_tril=self.cfg.R_chol
        )

    def _ensure_batch(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if len(x.shape) == 1:
            x = x[None, :]
        return x

    def _maybe_squeeze(self, x, squeeze):
        return x[0] if squeeze else x

    def sample_initial_particles(self, num_particles):
        x0 = self._prior_dist.sample(num_particles)
        return self.reflect_boundaries(x0)

    def transition_mean(self, x):
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)
        x_next = tf.matmul(x, self.cfg.F, transpose_b=True)
        return self._maybe_squeeze(x_next, squeeze)

    def transition(self, x, noise=True):
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)
        x_next = tf.matmul(x, self.cfg.F, transpose_b=True)

        if noise:
            eps = self._filter_process_noise_dist.sample(tf.shape(x)[0])
            x_next = x_next + eps

        x_next = self.reflect_boundaries(x_next)
        return self._maybe_squeeze(x_next, squeeze)

    def propagate_truth(self, x):
        """
        Propagate true state using Q_true.
        """
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)
        x_next = tf.matmul(x, self.cfg.F, transpose_b=True)
        eps = self._true_process_noise_dist.sample(tf.shape(x)[0])
        x_next = x_next + eps
        x_next = self.reflect_boundaries(x_next)
        return self._maybe_squeeze(x_next, squeeze)

    def propagate_particles(self, particles):
        """
        Propagate particles using Q_cov.
        """
        particles = self._ensure_batch(particles)
        x_next = tf.matmul(particles, self.cfg.F, transpose_b=True)
        eps = self._filter_process_noise_dist.sample(tf.shape(particles)[0])
        x_next = x_next + eps
        x_next = self.reflect_boundaries(x_next)
        return x_next

    def transition_log_prob(self, x_next, x_prev, use_truth_noise=False):
        """
        Log p(x_next | x_prev)
        use_truth_noise=False -> use filter covariance Q_cov
        use_truth_noise=True  -> use truth covariance Q_true
        """
        x_prev = self._ensure_batch(x_prev)
        x_next = self._ensure_batch(x_next)

        mean = tf.matmul(x_prev, self.cfg.F, transpose_b=True)

        chol = self.cfg.Q_true_chol if use_truth_noise else self.cfg.Q_chol
        chol_tiled = tf.tile(chol[None, :, :], [tf.shape(mean)[0], 1, 1])

        dist = tfd.MultivariateNormalTriL(
            loc=mean,
            scale_tril=chol_tiled
        )
        return dist.log_prob(x_next)

    def measurement_model(self, x):
        """
        Deterministic measurement mean.
        Supports shape [D] or [B,D].
        """
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)

        xt = tf.reshape(x, [-1, self.cfg.n_targets, 4])
        pos = xt[:, :, 0:2]   # [B,C,2]

        sensors = self.cfg.sensors_pos[None, None, :, :]  # [1,1,S,2]
        diff = pos[:, :, None, :] - sensors               # [B,C,S,2]
        dist = tf.norm(diff, axis=-1)                     # [B,C,S]

        signal = self.cfg.Amp / (dist + self.cfg.d0)
        z_mean = tf.reduce_sum(signal, axis=1)            # [B,S]

        return self._maybe_squeeze(z_mean, squeeze)

    # sample noisy measurement
    def sample_measurement(self, x):
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        z_mean = self._ensure_batch(self.measurement_model(x))
        noise = self._meas_noise_dist.sample(tf.shape(z_mean)[0])
        z = z_mean + noise
        return self._maybe_squeeze(z, squeeze)

    # measurement covariance
    def get_measurement_cov(self, x=None):
        return self.cfg.R_cov

    # measurement log-probability
    def measurement_log_prob(self, z, x):
        x = self._ensure_batch(x)
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        if len(z.shape) == 1:
            z = z[None, :]

        z_mean = self.measurement_model(x)
        chol = tf.tile(self.cfg.R_chol[None, :, :], [tf.shape(z_mean)[0], 1, 1])

        dist = tfd.MultivariateNormalTriL(
            loc=z_mean,
            scale_tril=chol
        )
        return dist.log_prob(z)

    # measurement Jacobian wrt state
    def measurement_jacobian(self, x):
        """
        Returns H = dz/dx with shape:
        [meas_dim, state_dim] for single x
        [B, meas_dim, state_dim] for batch input
        """
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)

        xt = tf.reshape(x, [-1, self.cfg.n_targets, 4])
        pos = xt[:, :, 0:2]  # [B,C,2]

        sensors = self.cfg.sensors_pos[None, None, :, :]  # [1,1,S,2]
        diff = pos[:, :, None, :] - sensors               # [B,C,S,2]
        dist = tf.norm(diff, axis=-1)                     # [B,C,S]
        dist_safe = tf.maximum(dist, 1e-4)

        # d/dpos [Amp / (dist + d0)] = -Amp * diff / ((dist+d0)^2 * dist)
        factor = -self.cfg.Amp / (((dist_safe + self.cfg.d0) ** 2) * dist_safe)
        dpos = factor[..., None] * diff                   # [B,C,S,2]

        cols = []
        zeros = tf.zeros_like(dpos[:, 0, :, 0])

        for c in range(self.cfg.n_targets):
            dx = dpos[:, c, :, 0]
            dy = dpos[:, c, :, 1]
            cols.extend([dx, dy, zeros, zeros])

        H = tf.stack(cols, axis=-1)  # [B,S,D]
        return self._maybe_squeeze(H, squeeze)

    # boundary reflection
    def reflect_boundaries(self, x):
        """
        Reflect positions into [min_pos, max_pos], and flip corresponding velocity.
        Supports [D] and [B,D].
        """
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)

        xt = tf.reshape(x, [-1, self.cfg.n_targets, 4])
        pos = xt[:, :, 0:2]
        vel = xt[:, :, 2:4]

        min_val = self.cfg.min_pos
        max_val = self.cfg.max_pos

        below = pos < min_val
        above = pos > max_val

        pos = tf.where(below, 2.0 * min_val - pos, pos)
        pos = tf.where(above, 2.0 * max_val - pos, pos)

        flip = tf.cast(tf.logical_or(below, above), tf.float32)
        vel = vel * (1.0 - 2.0 * flip)

        x_new = tf.concat([pos, vel], axis=-1)
        x_new = tf.reshape(x_new, [-1, self.cfg.state_dim])

        return self._maybe_squeeze(x_new, squeeze)
