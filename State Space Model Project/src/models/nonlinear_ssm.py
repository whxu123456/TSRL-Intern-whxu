import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class NonlinearSSMConfig:
    """
    Configuration class: Andrieu et al. (2010) Section 3.1 nonlinear state space model
    State equation: x_t = 0.5 x_{t-1} + 25 x_{t-1}/(1 + x_{t-1}²) + 8 cos(1.2(t-1)) + v_t, v_t ~ N(0, sigma_v²)
    Measurement equation: y_t = 0.05 x_t² + w_t, w_t ~ N(0, sigma_w²)
    """
    def __init__(self, T=100):
        # Time step length
        self.T = T
        self.dt = 1.0

        # State/measurement dimensions (scalar model, both 1D)
        self.state_dim = 1
        self.meas_dim = 1

        # Model core parameters
        self.alpha = 0.5    # Autoregressive coefficient
        self.beta = 25.0    # Nonlinear term coefficient
        self.gamma = 8.0    # Cosine term coefficient
        self.delta = 1.2    # Cosine term frequency
        self.kappa = 0.05   # Measurement nonlinear term coefficient

        # Noise parameters
        self.sigma_v = np.sqrt(10.0)  # Process noise standard deviation
        self.sigma_w = 1.0             # Measurement noise standard deviation
        self.sigma_v2 = self.sigma_v ** 2
        self.sigma_w2 = self.sigma_w ** 2

        # Noise covariance matrices (1D)
        self.Q_cov = tf.constant([[self.sigma_v2]], dtype=tf.float32)
        self.R_cov = tf.constant([[self.sigma_w2]], dtype=tf.float32)
        self.Q_chol = tf.linalg.cholesky(self.Q_cov)
        self.R_chol = tf.linalg.cholesky(self.R_cov)
        self.R_inv = tf.linalg.inv(self.R_cov)

        # Initial state distribution: x_0 ~ N(0, 5)
        self.x0_mean = tf.constant([0.0], dtype=tf.float32)
        self.x0_cov = tf.constant([[5.0]], dtype=tf.float32)
        self.P0 = tf.identity(self.x0_cov)
        self.P0_chol = tf.linalg.cholesky(self.P0)

        # Filter default particle count
        self.n_particles = 500


class NonlinearSSMModel:
    """
    Nonlinear state space model core class, implements all interfaces required for Li (2017) PF-PF
    """
    def __init__(self, config: NonlinearSSMConfig):
        self.cfg = config
        # Distribution definitions
        self._prior_dist = tfd.MultivariateNormalTriL(
            loc=self.cfg.x0_mean,
            scale_tril=self.cfg.P0_chol
        )
        self._process_noise_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.cfg.state_dim, dtype=tf.float32),
            scale_tril=self.cfg.Q_chol
        )
        self._meas_noise_dist = tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.cfg.meas_dim, dtype=tf.float32),
            scale_tril=self.cfg.R_chol
        )

    def _ensure_batch(self, x):
        """Ensure input has batch dimension [B, state_dim]"""
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if len(x.shape) == 1:
            x = x[None, :]
        return x

    def _maybe_squeeze(self, x, squeeze):
        """Squeeze batch dimension on demand"""
        return x[0] if squeeze else x

    def sample_initial_particles(self, num_particles):
        """Sample initial particles"""
        return self._prior_dist.sample(num_particles)

    def transition_mean(self, x, t):
        """
        State transition mean (without noise)
        Args:
            x: State, shape [B, state_dim] or [state_dim]
            t: Current time step (starting from 1, corresponds to cos(1.2*(t-1)))
        Returns:
            x_mean: Transitioned mean, shape matches input
        """
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)
        x_scalar = x[:, 0]

        # State equation core computation
        term1 = self.cfg.alpha * x_scalar
        term2 = self.cfg.beta * x_scalar / (1.0 + x_scalar ** 2)
        term3 = self.cfg.gamma * tf.cos(self.cfg.delta * (t - 1))
        x_mean = term1 + term2 + term3

        x_mean = tf.reshape(x_mean, [-1, self.cfg.state_dim])
        return self._maybe_squeeze(x_mean, squeeze)

    def propagate_particles(self, x_prev, t):
        """
        Particle propagation with noise (prior prediction)
        Args:
            x_prev: Previous time step particles, shape [N, state_dim]
            t: Current time step
        Returns:
            x_pred: Propagated prior particles, shape [N, state_dim]
        """
        x_prev = self._ensure_batch(x_prev)
        x_mean = self.transition_mean(x_prev, t)
        eps = self._process_noise_dist.sample(tf.shape(x_prev)[0])
        x_pred = x_mean + eps
        return x_pred

    def propagate_truth(self, x_prev, t):
        """State propagation for generating true trajectory"""
        return self.propagate_particles(x_prev, t)

    def transition_log_prob(self, x_next, x_prev, t):
        """
        Transition log probability log p(x_next | x_prev)
        Args:
            x_next: Current state, shape [B, state_dim]
            x_prev: Previous time step state, shape [B, state_dim]
            t: Current time step
        Returns:
            log_prob: Log probability, shape [B]
        """
        x_prev = self._ensure_batch(x_prev)
        x_next = self._ensure_batch(x_next)
        mean = self.transition_mean(x_prev, t)

        chol_tiled = tf.tile(self.cfg.Q_chol[None, :, :], [tf.shape(mean)[0], 1, 1])
        dist = tfd.MultivariateNormalTriL(loc=mean, scale_tril=chol_tiled)
        return dist.log_prob(x_next)

    def measurement_model(self, x):
        """
        Measurement model mean (without noise)
        Args:
            x: State, shape [B, state_dim] or [state_dim]
        Returns:
            z_mean: Measurement mean, shape matches input [B, meas_dim] or [meas_dim]
        """
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)
        x_scalar = x[:, 0]
        z_mean = self.cfg.kappa * (x_scalar ** 2)
        z_mean = tf.reshape(z_mean, [-1, self.cfg.meas_dim])
        return self._maybe_squeeze(z_mean, squeeze)

    def sample_measurement(self, x):
        """Sample observation with noise"""
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        z_mean = self._ensure_batch(self.measurement_model(x))
        noise = self._meas_noise_dist.sample(tf.shape(z_mean)[0])
        z = z_mean + noise
        return self._maybe_squeeze(z, squeeze)

    def measurement_log_prob(self, z, x):
        """
        Measurement log probability log p(z | x)
        Args:
            z: Observation, shape [meas_dim]
            x: State, shape [B, state_dim]
        Returns:
            log_prob: Log probability, shape [B]
        """
        x = self._ensure_batch(x)
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        if len(z.shape) == 1:
            z = z[None, :]

        z_mean = self.measurement_model(x)
        chol_tiled = tf.tile(self.cfg.R_chol[None, :, :], [tf.shape(z_mean)[0], 1, 1])
        dist = tfd.MultivariateNormalTriL(loc=z_mean, scale_tril=chol_tiled)
        return dist.log_prob(z)

    def measurement_jacobian(self, x):
        """
        Measurement model Jacobian matrix H = dz/dx
        Args:
            x: State, shape [B, state_dim] or [state_dim]
        Returns:
            H: Jacobian matrix, shape [B, meas_dim, state_dim] or [meas_dim, state_dim]
        """
        squeeze = len(tf.convert_to_tensor(x).shape) == 1
        x = self._ensure_batch(x)
        x_scalar = x[:, 0]
        # 1D model Jacobian: dh/dx = 2 * kappa * x
        dh_dx = 2.0 * self.cfg.kappa * x_scalar
        H = tf.reshape(dh_dx, [-1, self.cfg.meas_dim, self.cfg.state_dim])
        return self._maybe_squeeze(H, squeeze)

    def generate_true_trajectory(self):
        """Generate true state trajectory and observation sequence"""
        x_true = np.zeros((self.cfg.T, self.cfg.state_dim), dtype=np.float32)
        y_obs = np.zeros((self.cfg.T, self.cfg.meas_dim), dtype=np.float32)

        # Initial state
        x_true[0] = self._prior_dist.sample().numpy()
        y_obs[0] = self.sample_measurement(x_true[0]).numpy()

        # Recursive generation
        for t in range(1, self.cfg.T):
            x_true[t] = self.propagate_truth(x_true[t-1], t).numpy()
            y_obs[t] = self.sample_measurement(x_true[t]).numpy()

        return x_true, y_obs