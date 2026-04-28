import unittest
import numpy as np
import tensorflow as tf
from src.models.stochastic_volatility import SVModel
from src.filters.sv_particle_flow_filter import SVParticleFlowFilter


class TestSVModel(unittest.TestCase):
    """
    Simplified tests for SVModel (full coverage completed in prior tests).
    Only validates basic data generation for filter integration context.
    """

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)
        self.T = 50
        self.model = SVModel(alpha=0.9, sigma=1.0, beta=0.5, T=self.T)

    def test_data_generation_basic(self):
        """
        Verify generated data has valid shape and finite values.
        """
        x_true, y_obs = self.model.generate_data()
        self.assertEqual(x_true.shape, (self.T,))
        self.assertEqual(y_obs.shape, (self.T,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(x_true)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(y_obs)))


class TestSVParticleFlowFilter(unittest.TestCase):
    """
    Unit tests for SVParticleFlowFilter.
    """

    def setUp(self):
        """
        Initialize filter instances for all supported methods and test parameters.
        """
        np.random.seed(42)
        tf.random.set_seed(42)

        self.model = SVModel(T=20)
        self.num_particles = 50
        self.num_flow_steps = 10
        self.test_observation = tf.constant(0.5, dtype=tf.float32)

        # Initialize filters for all methods
        self.filters = {
            'EDH_Exact': SVParticleFlowFilter(self.model, self.num_particles, self.num_flow_steps, 'EDH_Exact'),
            'LEDH_Exact': SVParticleFlowFilter(self.model, self.num_particles, self.num_flow_steps, 'LEDH_Exact'),
            'EDH_Log': SVParticleFlowFilter(self.model, self.num_particles, self.num_flow_steps, 'EDH_Log'),
            'Kernel': SVParticleFlowFilter(self.model, self.num_particles, self.num_flow_steps, 'Kernel')
        }

    def test_filter_initialization(self):
        """
        Test __init__ method: validates particle shape, weights, parameters and attributes.
        """
        pf = self.filters['EDH_Exact']
        # Check particle shape
        self.assertEqual(pf.particles.shape, (self.num_particles, 1))
        # Check uniform weights
        self.assertEqual(pf.weights.shape, (self.num_particles,))
        self.assertTrue(np.allclose(pf.weights.numpy(), 1 / self.num_particles))
        # Check SV-specific attributes
        self.assertEqual(pf.obs_noise_mean, -1.27)
        self.assertEqual(pf.method, 'EDH_Exact')
        # Check diagnostic lists are empty
        self.assertEqual(len(pf.flow_mags), 0)
        self.assertEqual(len(pf.condition_numbers), 0)

    def test_predict_step(self):
        """
        Test predict method: validates state propagation and shape preservation.
        """
        pf = self.filters['EDH_Exact']
        original_particles = tf.identity(pf.particles)

        pf.predict()

        # Particles are updated
        self.assertFalse(np.allclose(original_particles.numpy(), pf.particles.numpy()))
        # Shape remains unchanged
        self.assertEqual(pf.particles.shape, (self.num_particles, 1))
        # Diagnostics are reset
        self.assertEqual(len(pf.flow_mags), 0)

    def test_flow_edh_exact(self):
        """
        Test flow_edh_exact method: standalone EDH flow without log transform.
        Validates particle update and numerical stability.
        """
        pf = self.filters['EDH_Exact']
        original_particles = tf.identity(pf.particles)

        pf.flow_edh_exact(self.test_observation)

        # Particles are modified
        self.assertFalse(np.allclose(original_particles.numpy(), pf.particles.numpy()))
        # No NaN/Inf values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pf.particles)))
        # Diagnostics are recorded
        self.assertEqual(len(pf.flow_mags), self.num_flow_steps)

    def test_flow_ledh_exact(self):
        """
        Test flow_ledh_exact method: standalone LEDH flow without log transform.
        Validates per-particle linearization and numerical stability.
        """
        pf = self.filters['LEDH_Exact']
        original_particles = tf.identity(pf.particles)

        pf.flow_ledh_exact(self.test_observation)

        self.assertFalse(np.allclose(original_particles.numpy(), pf.particles.numpy()))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pf.particles)))
        self.assertEqual(len(pf.flow_mags), self.num_flow_steps)

    def test_flow_edh_ledh_log_trans(self):
        """
        Test flow_edh_ledh_log_trans method: flow with log(y²) observation transform.
        Validates linear observation model flow and stability.
        """
        pf = self.filters['EDH_Log']
        original_particles = tf.identity(pf.particles)

        pf.flow_edh_ledh_log_trans(self.test_observation)

        self.assertFalse(np.allclose(original_particles.numpy(), pf.particles.numpy()))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pf.particles)))
        self.assertEqual(len(pf.flow_mags), self.num_flow_steps)

    def test_rbf_kernel(self):
        """
        Test rbf_kernel method: validates kernel matrix and gradient shape/properties.
        """
        pf = self.filters['Kernel']
        test_particles = tf.random.normal((self.num_particles, 1))

        K, grad_K = pf.rbf_kernel(test_particles)

        # Check shape
        self.assertEqual(K.shape, (self.num_particles, self.num_particles))
        self.assertEqual(grad_K.shape, (self.num_particles, self.num_particles))
        # Kernel diagonal is 1 (RBF property)
        diag_vals = tf.linalg.diag_part(K)
        self.assertTrue(np.allclose(diag_vals.numpy(), 1.0, atol=1e-5))
        # Finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(K)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(grad_K)))

    def test_flow_kernel(self):
        """
        Test flow_kernel method: standalone kernel-based particle flow.
        Validates gradient tape computation and particle update.
        """
        pf = self.filters['Kernel']
        original_particles = tf.identity(pf.particles)

        pf.flow_kernel(self.test_observation)

        self.assertFalse(np.allclose(original_particles.numpy(), pf.particles.numpy()))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(pf.particles)))
        self.assertEqual(len(pf.flow_mags), self.num_flow_steps)

    def test_update_dispatcher(self):
        """
        Test update method: validates correct flow method dispatching for all 4 modes.
        Ensures no errors and stable outputs for all supported methods.
        """
        for method_name, pf in self.filters.items():
            with self.subTest(method=method_name):
                original_particles = tf.identity(pf.particles)
                pf.update(self.test_observation)

                # Verify particles are updated
                self.assertFalse(np.allclose(original_particles.numpy(), pf.particles.numpy()))
                self.assertTrue(tf.reduce_all(tf.math.is_finite(pf.particles)))

    def test_full_filter_cycle_basic(self):
        """
        Basic integration test: runs predict + update for 2 time steps.
        """
        pf = self.filters['Kernel']
        x_true, y_obs = self.model.generate_data()

        for t in range(2):
            pf.predict()
            pf.update(y_obs[t])
            # Valid estimated mean
            mean_estimate = tf.reduce_mean(pf.particles).numpy()
            self.assertFalse(np.isnan(mean_estimate))


if __name__ == '__main__':
    unittest.main()