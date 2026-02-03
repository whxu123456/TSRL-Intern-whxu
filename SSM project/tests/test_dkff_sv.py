import unittest
import numpy as np
import tensorflow as tf
from src.models.stochastic_volatility import SVModel
from src.filters.particle_flow_filter import SVParticleFlowFilter

class TestSVModel(unittest.TestCase):
    """
    Test Stochastic Volatility Model (The same as the part in test_nonlinear.py)
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

    def test_generate_data(self):
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

    def test_log_likelihood_gradients(self):
        """
        Test log likelihood gradients.
        """
        y_obs = tf.constant(1.0, dtype=tf.float32)
        x_particles = tf.Variable([[0.5]], dtype=tf.float32)

        with tf.GradientTape() as tape:
            log_prob = self.model.log_likelihood(y_obs, x_particles)

        grad = tape.gradient(log_prob, x_particles)

        # Check whether gradient is None.
        self.assertIsNotNone(grad)
        # Check whether gradient is finite.
        self.assertFalse(tf.math.is_nan(grad))


class TestSVParticleFlowFilter(unittest.TestCase):
    """
    Test Particle Flow Filter under SVModel
    """

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

        self.T = 20
        self.model = SVModel(T=self.T)
        self.num_particles = 100
        self.num_steps = 10

        # Initialize Filter under different methods
        self.pf_edh = SVParticleFlowFilter(
            self.model, num_particles=self.num_particles, num_steps=self.num_steps, method='EDH'
        )
        self.pf_kernel = SVParticleFlowFilter(
            self.model, num_particles=self.num_particles, num_steps=self.num_steps, method='Kernel'
        )

    def test_init_particles(self):
        """
        Test the intial particles.
        """

        self.assertEqual(self.pf_edh.particles.shape, (self.num_particles, 1)) # [N, 1]

        # Check whether the sample variance = (sigma^2 / (1-alpha^2))
        expected_std = self.model.sigma / np.sqrt(1 - self.model.alpha ** 2)
        actual_std = tf.math.reduce_std(self.pf_edh.particles).numpy()
        self.assertTrue(abs(actual_std - expected_std) / expected_std < 0.25)

    def test_predict_step(self):
        """
        Test the prediction step.
        """

        old_particles = tf.identity(self.pf_edh.particles)
        self.pf_edh.predict()
        new_particles = self.pf_edh.particles
        self.assertFalse(np.allclose(old_particles.numpy(), new_particles.numpy()))
        # Maintain the same shape.
        self.assertEqual(new_particles.shape, (self.num_particles, 1))

    def test_compute_covariance(self):
        """
        Test the covariance computation.
        """

        # sample data: [[1], [2], [3]]
        data = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
        # Change temporarily for the test.
        self.pf_edh.N = 3
        cov = self.pf_edh.compute_covariance(data)

        # var([1,2,3]) = 1.0
        self.assertAlmostEqual(cov.numpy()[0, 0], 1.0, places=5)

        # restore N
        self.pf_edh.N = self.num_particles

    def test_rbf_kernel(self):
        """
        Test RBF kernel.
        """

        X = tf.random.normal((self.num_particles, 1))
        K, grad_K = self.pf_kernel.rbf_kernel(X)

        # [N, N]
        self.assertEqual(K.shape, (self.num_particles, self.num_particles))
        self.assertEqual(grad_K.shape, (self.num_particles, self.num_particles))

        # The diagonal elements of the kernel should be 1.
        diag = tf.linalg.diag_part(K)
        self.assertTrue(np.allclose(diag.numpy(), 1.0, atol=1e-5))

    def test_update_edh_integration(self):
        """
        Test the EDH method.
        """

        y_obs = tf.constant(0.5, dtype=tf.float32)
        try:
            self.pf_edh.update(y_obs)
        except Exception as e:
            self.fail(f"EDH update failed with error: {e}")
        self.assertFalse(np.any(np.isnan(self.pf_edh.particles.numpy())))

    def test_update_kernel_integration(self):
        """
        Test the kernel method.
        """

        y_obs = tf.constant(-0.5, dtype=tf.float32)

        try:
            self.pf_kernel.update(y_obs)
        except Exception as e:
            self.fail(f"Kernel update failed with error: {e}")

        self.assertEqual(len(self.pf_kernel.flow_mags), self.num_steps)
        self.assertFalse(np.any(np.isnan(self.pf_kernel.particles.numpy())))

    def test_full_trajectory_loop(self):
        """
        simulate the loop the main_dkff_sv.py.
        """

        x_true, y_obs = self.model.generate_data()

        # Run the first 3 steps.
        for t in range(3):
            self.pf_kernel.predict()
            self.pf_kernel.update(y_obs[t])

            mean_est = tf.reduce_mean(self.pf_kernel.particles).numpy()
            self.assertFalse(np.isnan(mean_est))


if __name__ == '__main__':
    unittest.main()