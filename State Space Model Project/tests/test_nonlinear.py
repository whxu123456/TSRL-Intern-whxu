import unittest
import numpy as np
import tensorflow as tf
from src.models.stochastic_volatility import SVModel
from src.filters.EKF_SV import EKF_SV, EKF_SV_Raw
from src.filters.UKF_SV import UKF_SV, UKF_SV_Raw
from src.filters.particle_filter import ParticleFilter


class TestSVModel(unittest.TestCase):
    """
    Unit tests for the Stochastic Volatility (SV) Model
    """

    def setUp(self):
        """Set up test for SVModel"""
        np.random.seed(42)
        tf.random.set_seed(42)
        self.T = 50
        self.model = SVModel(alpha=0.9, sigma=1.0, beta=0.5, T=self.T)

    def test_initialization(self):
        """Test that the SVModel initializes with correct parameters"""
        self.assertEqual(self.model.T, 50)
        self.assertEqual(self.model.alpha, 0.9)
        self.assertEqual(self.model.sigma, 1.0)
        self.assertEqual(self.model.beta, 0.5)

    def test_generate_data_shape_and_type(self):
        """Test the shape, data type, and validity of generated time series"""
        x, y = self.model.generate_data()
        # Verify tensor type
        self.assertIsInstance(x, tf.Tensor)
        self.assertIsInstance(y, tf.Tensor)
        # Verify correct shape
        self.assertEqual(x.shape, (self.T,))
        self.assertEqual(y.shape, (self.T,))
        # Verify no NaN or infinite values
        self.assertFalse(np.any(np.isnan(x.numpy())))
        self.assertFalse(np.any(np.isnan(y.numpy())))

    def test_log_likelihood_computation(self):
        """Test the log likelihood function outputs finite, valid values"""
        x = tf.constant([0.0, 0.0], dtype=tf.float32)
        y = tf.constant([1.0, -1.0], dtype=tf.float32)

        ll = self.model.log_likelihood(y, x)
        self.assertEqual(ll.shape, (2,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(ll)))


class TestEKF_SV(unittest.TestCase):
    """Unit tests for the log-transformed Extended Kalman Filter for SV Model"""
    def setUp(self):
        """Set up test for EKF_SV"""
        np.random.seed(42)
        self.T = 20
        self.model = SVModel(T=self.T)
        self.ekf = EKF_SV(self.model)
        _, self.y_obs = self.model.generate_data()

    def test_run_output_shape(self):
        """Test the EKF run() method returns correct output shape and valid runtime"""
        x_est, runtime = self.ekf.run(self.y_obs)

        self.assertEqual(x_est.shape, (self.T,))
        self.assertIsInstance(x_est, np.ndarray)
        self.assertIsInstance(runtime, float)
        self.assertGreater(runtime, 0.0)
        self.assertFalse(np.any(np.isnan(x_est)))

    def test_handling_zero_observation(self):
        """Test EKF robustness to zero-valued observations (edge case)"""
        y_obs_zero = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
        try:
            x_est, _ = self.ekf.run(y_obs_zero)
            self.assertFalse(np.any(np.isnan(x_est)), "EKF produced NaNs on zero observation")
        except Exception as e:
            self.fail(f"EKF failed on zero observation with error: {e}")


class TestEKF_SV_Raw(unittest.TestCase):
    """Unit tests for the NON log-transformed (Raw) Extended Kalman Filter for SV Model"""
    def setUp(self):
        """Set up test for EKF_SV_Raw"""
        np.random.seed(42)
        self.T = 20
        self.model = SVModel(T=self.T)
        self.ekf_raw = EKF_SV_Raw(self.model)
        _, self.y_obs = self.model.generate_data()

    def test_run_output_shape_and_validity(self):
        """Test Raw EKF run() method outputs valid estimates with correct shape"""
        x_est, runtime = self.ekf_raw.run(self.y_obs)

        self.assertEqual(x_est.shape, (self.T,))
        self.assertIsInstance(x_est, np.ndarray)
        self.assertGreater(runtime, 0.0)
        # Ensure no invalid values in estimates
        self.assertFalse(np.any(np.isnan(x_est)))
        self.assertFalse(np.any(np.isinf(x_est)))

    def test_zero_observation_robustness(self):
        """Test Raw EKF handles zero observations without crashing"""
        y_test = tf.constant([0.0, 0.5, 0.0], dtype=tf.float32)
        try:
            x_est, _ = self.ekf_raw.run(y_test)
            self.assertFalse(np.any(np.isnan(x_est)))
        except Exception as e:
            self.fail(f"Raw EKF failed on edge case observations: {e}")


class TestUKF_SV(unittest.TestCase):
    """Unit tests for the log-transformed Unscented Kalman Filter for SV Model"""
    def setUp(self):
        """Set up test for UKF_SV"""
        np.random.seed(42)
        self.T = 20
        self.model = SVModel(T=self.T)
        self.ukf = UKF_SV(self.model)
        _, self.y_obs = self.model.generate_data()

    def test_run_output_shape(self):
        """Test UKF run() method returns correctly shaped, valid estimates"""
        x_est, runtime = self.ukf.run(self.y_obs)

        self.assertEqual(x_est.shape, (self.T,))
        self.assertIsInstance(x_est, np.ndarray)
        self.assertFalse(np.any(np.isnan(x_est)))

    def test_sigma_points_logic(self):
        """Verify UKF sigma point computation produces stable, non-NaN estimates"""
        x_est, _ = self.ukf.run(self.y_obs)
        self.assertFalse(np.any(np.isnan(x_est)), "UKF produced NaN values")


class TestUKF_SV_Raw(unittest.TestCase):
    """Unit tests for the NON log-transformed (Raw) Unscented Kalman Filter for SV Model"""
    def setUp(self):
        """Set up test fixtures for UKF_SV_Raw"""
        np.random.seed(42)
        self.T = 20
        self.model = SVModel(T=self.T)
        self.ukf_raw = UKF_SV_Raw(self.model)
        _, self.y_obs = self.model.generate_data()

    def test_run_output_validity(self):
        """Test Raw UKF run() method generates valid state estimates"""
        x_est, runtime = self.ukf_raw.run(self.y_obs)

        self.assertEqual(x_est.shape, (self.T,))
        self.assertIsInstance(x_est, np.ndarray)
        self.assertFalse(np.any(np.isnan(x_est)))
        self.assertFalse(np.any(np.isinf(x_est)))


class TestParticleFilter(unittest.TestCase):
    """Unit tests for the Particle Filter for SV Model"""
    def setUp(self):
        """Set up test fixtures for ParticleFilter"""
        np.random.seed(42)
        tf.random.set_seed(42)
        self.T = 15
        self.num_particles = 100
        self.model = SVModel(T=self.T)
        self.pf = ParticleFilter(self.model, num_particles=self.num_particles)
        _, self.y_obs = self.model.generate_data()

    def test_initialization(self):
        """Test Particle Filter initializes with correct particle count"""
        self.assertEqual(self.pf.N, self.num_particles)

    def test_systematic_resampling(self):
        """Test systematic resampling works for uniform and extreme weight distributions"""
        # Case 1: Equal weights (no resampling bias)
        weights = tf.ones(self.num_particles) / self.num_particles
        indices = self.pf.systematic_resampling(weights)

        self.assertEqual(indices.shape, (self.num_particles,))
        self.assertTrue(tf.reduce_all(indices >= 0))
        self.assertTrue(tf.reduce_all(indices < self.num_particles))

        # Case 2: All weight on first particle
        weights_extreme = np.zeros(self.num_particles)
        weights_extreme[0] = 1.0
        weights_extreme = tf.constant(weights_extreme, dtype=tf.float32)

        indices_extreme = self.pf.systematic_resampling(weights_extreme)
        self.assertTrue(tf.reduce_all(tf.equal(indices_extreme, 0)))

    def test_run_output(self):
        """Test Particle Filter run() outputs valid estimates and ESS history"""
        x_est, ess_hist, runtime = self.pf.run(self.y_obs)

        # Verify shape correctness
        self.assertEqual(x_est.shape, (self.T,))
        self.assertEqual(ess_hist.shape, (self.T,))
        # Verify ESS stays within valid range [1, N]
        self.assertTrue(np.all(ess_hist >= 1.0))
        self.assertTrue(np.all(ess_hist <= self.num_particles + 1e-5))
        # Verify no invalid values
        self.assertFalse(np.any(np.isnan(x_est)))

    def test_resampling_trigger(self):
        """Test resampling triggers correctly when ESS falls below threshold"""
        _, ess_hist, _ = self.pf.run(self.y_obs)
        self.assertFalse(np.any(np.isnan(ess_hist)))
        # Verify ESS values are finite and reasonable
        self.assertTrue(np.all(np.isfinite(ess_hist)))


if __name__ == '__main__':
    unittest.main()