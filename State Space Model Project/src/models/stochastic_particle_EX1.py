import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class Dai22Example1Model:
    """
    Static model for Dai22 Example 1.
      - sample_initial_particles
      - propagate_particles
      - measurement_model
      - get_measurement_cov
      - sample_measurement
      - get_prior_dist
    """

    def __init__(self):
        # True hidden state
        self.x_true = tf.constant([4.0, 4.0], dtype=tf.float32)

        # Sensors location
        self.sensor_locs = tf.constant(
            [[3.5, 0.0],
             [-3.5, 0.0]],
            dtype=tf.float32
        )

        # Prior p0(x)
        self.prior_mean = tf.constant([3.0, 5.0], dtype=tf.float32)
        self.prior_cov = tf.linalg.diag([1000.0, 2.0])

        # Measurement noise covariance
        self.R = tf.linalg.diag([0.04, 0.04])

        # Flow diffusion covariance
        self.Q_flow = tf.linalg.diag([4.0, 0.4])

        # Parameter used in the beta(lambda) BVP
        self.mu = 0.2

        self.state_dim = 2
        self.meas_dim = 2

    def measurement_model(self, x):
        """
        Bearing measurement model.
        x : Shape (..., 2)
        Returns: angles : Shape (..., 2)
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        # (..., 1, 2)
        x_expanded = tf.expand_dims(x, axis=-2)
        # (..., 2, 2)
        diff = x_expanded - self.sensor_locs
        # (..., 2)
        angles = tf.math.atan2(diff[..., 1], diff[..., 0])
        return angles

    def get_measurement_cov(self, x=None):
        """
        Return measurement covariance R.
        """
        return self.R

    def get_prior_dist(self):
        """
        Return the prior distribution p0(x).
        """
        return tfd.MultivariateNormalFullCovariance(
            loc=self.prior_mean,
            covariance_matrix=self.prior_cov
        )

    def sample_initial_particles(self, num_particles):
        """
        Sample particles from the static prior.
        """
        return self.get_prior_dist().sample(num_particles)

    def propagate_particles(self, particles):
        """
        Static problem: no state dynamics.
        """
        return tf.identity(particles)

    def sample_measurement(self, x_true=None):
        """
        Generate one noisy measurement from the true state.
        """
        if x_true is None:
            x_true = self.x_true

        clean_z = self.measurement_model(x_true)

        noise_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.meas_dim, dtype=tf.float32),
            scale_diag=tf.sqrt(tf.linalg.diag_part(self.R))
        )
        return clean_z + noise_dist.sample()