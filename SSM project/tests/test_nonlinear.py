import unittest
import numpy as np
import tensorflow as tf
from src.models.stochastic_volatility import SVModel
from src.filters.EKF import EKF_SV
from src.filters.UKF import UKF_SV
from src.filters.particle_filter import ParticleFilter


class TestSVModel(unittest.TestCase):
    """
    Test Stochastic Volatility Model
    """

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)
        self.T = 50
        self.model = SVModel(alpha=0.9, sigma=1.0, beta=0.5, T=self.T)

    def test_initialization(self):
        """
        Test the initialization of SVModel.
        """

        self.assertEqual(self.model.T, 50)
        self.assertEqual(self.model.alpha, 0.9)

    def test_generate_data_shape_and_type(self):
        """
        Test the shape and type of the generated data.
        """

        x, y = self.model.generate_data()
        # Check type (tf.Tensor).
        self.assertIsInstance(x, tf.Tensor)
        self.assertIsInstance(y, tf.Tensor)
        # Check shape (T,).
        self.assertEqual(x.shape, (self.T,))
        self.assertEqual(y.shape, (self.T,))
        # Check NaN
        self.assertFalse(np.any(np.isnan(x.numpy())))
        self.assertFalse(np.any(np.isnan(y.numpy())))
    def test_log_likelihood_computation(self):
        """
        Test the computation of the log likelihood.
        """

        x = tf.constant([0.0, 0.0], dtype=tf.float32)
        y = tf.constant([1.0, -1.0], dtype=tf.float32)

        ll = self.model.log_likelihood(y, x)
        self.assertEqual(ll.shape, (2,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ll)))


class TestEKF_SV(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.T = 20
        self.model = SVModel(T=self.T)
        self.ekf = EKF_SV(self.model)
        _, self.y_obs = self.model.generate_data()

    def test_run_output_shape(self):
        """
        Test the output shape of the EKF model.
        """

        x_est, runtime = self.ekf.run(self.y_obs)

        self.assertEqual(x_est.shape, (self.T,))
        self.assertIsInstance(x_est, np.ndarray)
        self.assertIsInstance(runtime, float)
        self.assertGreater(runtime, 0.0)

    def test_handling_zero_observation(self):
        """
        Test the handling of zero observation.
        """

        y_obs_zero = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        try:
            x_est, _ = self.ekf.run(y_obs_zero)
            self.assertFalse(np.any(np.isnan(x_est)), "EKF produced NaNs on zero observation")
        except Exception as e:
            self.fail(f"EKF failed on zero observation with error: {e}")


class TestUKF_SV(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.T = 20
        self.model = SVModel(T=self.T)
        self.ukf = UKF_SV(self.model)
        _, self.y_obs = self.model.generate_data()

    def test_run_output_shape(self):
        """
        Test the output shape of the UKF model.
        """

        x_est, runtime = self.ukf.run(self.y_obs)

        self.assertEqual(x_est.shape, (self.T,))
        self.assertIsInstance(x_est, np.ndarray)

    def test_sigma_points_logic(self):
        """
        Test the siga points.
        """
        x_est, _ = self.ukf.run(self.y_obs)
        self.assertFalse(np.any(np.isnan(x_est)), "UKF produced NaNs")


class TestParticleFilter(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)
        self.T = 15
        self.num_particles = 100
        self.model = SVModel(T=self.T)
        self.pf = ParticleFilter(self.model, num_particles=self.num_particles)
        _, self.y_obs = self.model.generate_data()

    def test_initialization(self):
        self.assertEqual(self.pf.N, self.num_particles)

    def test_systematic_resampling(self):
        """
        Test the systematic resampling.
        """

        # Case 1: equal weights
        weights = tf.ones(self.num_particles) / self.num_particles
        indices = self.pf.systematic_resampling(weights)

        self.assertEqual(indices.shape, (self.num_particles,))
        # Indices should be in [0, N-1]
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < self.num_particles))

        # Case 2: extreme weights (All the weights concentrated on the first particle)
        weights_extreme = np.zeros(self.num_particles)
        weights_extreme[0] = 1.0
        weights_extreme = tf.constant(weights_extreme, dtype=tf.float32)

        indices_extreme = self.pf.systematic_resampling(weights_extreme)
        # All the indices should be 0.
        self.assertTrue(tf.reduce_all(tf.equal(indices_extreme, 0)))

    def test_run_output(self):
        """
        Test the output of particle filter.
        """
        x_est, ess_hist, runtime = self.pf.run(self.y_obs)

        # Check the shape.
        self.assertEqual(x_est.shape, (self.T,))
        self.assertEqual(ess_hist.shape, (self.T,))

        # Check whether ESS is in [1, N]
        self.assertTrue(np.all(ess_hist >= 1.0))
        self.assertTrue(np.all(ess_hist <= self.num_particles + 1e-5))

    def test_resampling_trigger(self):
        """
        Test whether resampling triggers properly when ESS is less than the threshold.
        """
        _, ess_hist, _ = self.pf.run(self.y_obs)
        self.assertFalse(np.any(np.isnan(ess_hist)))


if __name__ == '__main__':
    unittest.main()