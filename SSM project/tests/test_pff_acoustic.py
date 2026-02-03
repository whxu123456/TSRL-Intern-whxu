import unittest
import numpy as np
import tensorflow as tf
from src.models.acoustic import AcousticConfig, AcousticModel
from src.filters.EKF import EKF_Predictor
from src.filters.particle_flow_filter import AcousticPFPF


class TestAcousticModel(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        np.random.seed(42)
        self.cfg = AcousticConfig()
        self.cfg.n_particles = 100
        self.model = AcousticModel(self.cfg)

    def test_config_integrity(self):
        """
        Test the integrity of parameters.
        """

        self.assertEqual(self.cfg.state_dim, 16)
        self.assertEqual(self.cfg.meas_dim, 25)
        self.assertEqual(self.cfg.sensors_pos.shape, (25, 2))

    def test_transition_shape(self):
        """
        Test the shape of transition matrix.
        """

        # Batch size = 5
        x = tf.random.normal((5, self.cfg.state_dim))
        x_new = self.model.transition(x, noise=True)

        self.assertEqual(x_new.shape, (5, self.cfg.state_dim))
        self.assertFalse(np.allclose(x.numpy(), x_new.numpy()))

    def test_measurement_shape(self):
        """
        Test the shape of measurement matrix.
        """

        # Batch size = 5
        x = tf.random.normal((5, self.cfg.state_dim))
        z = self.model.measurement(x)

        self.assertEqual(z.shape, (5, self.cfg.meas_dim))

    def test_measurement_jacobian_numerical(self):
        """
        Test the calculation of Jacobian matrix.
        """

        # Batch size = 2
        x = tf.random.normal((2, self.cfg.state_dim))

        # 1. Calculate the Jacobian by measurement_jacobian function [batch, meas_dim, state_dim]
        H_analytic = self.model.measurement_jacobian(x)

        # 2. Calculate the Jacobian by tf.GradientTape
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = self.model.measurement(x)
        # [2, 25, 16]
        H_autodiff = tape.batch_jacobian(z, x)
        np.testing.assert_allclose(
            H_analytic.numpy(),
            H_autodiff.numpy(),
            rtol=1e-3, atol=1e-3,
            err_msg="Analytical Jacobian does not match Auto-diff Jacobian"
        )

    def test_limit_boundary(self):
        """
        Test the limit boundary function.
        """

        # Create a state outside the boundary (-10,50).
        # The boundary is [-5, 45].
        # target 0 -> x:0, y:1, vx:2, vy:3
        x_bad = np.zeros(self.cfg.state_dim, dtype=np.float32)
        x_bad[0] = -10.0  # x < -5
        x_bad[2] = 1.0  # vx positive
        x_bad[1] = 50.0  # y > 45
        x_bad[3] = 1.0  # vy positive

        x_bad_tf = tf.constant(x_bad)
        x_limited = self.model.limit_boundary(x_bad_tf)
        x_limited_np = x_limited.numpy()
        # -10 -> -5 + (-5 - (-10)) = 0
        self.assertAlmostEqual(x_limited_np[0], 0.0)
        self.assertAlmostEqual(x_limited_np[2], -1.0)
        # 50 -> 45 - (50 - 45) = 40
        self.assertAlmostEqual(x_limited_np[1], 40.0)
        self.assertAlmostEqual(x_limited_np[3], -1.0)


class TestEKF_Predictor(unittest.TestCase):
    def setUp(self):
        self.cfg = AcousticConfig()
        self.model = AcousticModel(self.cfg)
        self.ekf = EKF_Predictor(self.cfg, self.model)

    def test_initialization(self):
        self.assertEqual(self.ekf.P.shape, (self.cfg.state_dim, self.cfg.state_dim))
        # x should be reshaped [1, 16].
        self.assertEqual(self.ekf.x.shape, (1, self.cfg.state_dim))

    def test_predict_step(self):
        """
        Test the prediction step.
        """

        P_old = self.ekf.P
        x_old = self.ekf.x
        P_pred = self.ekf.predict()
        self.assertEqual(P_pred.shape, (self.cfg.state_dim, self.cfg.state_dim))
        self.assertTrue(np.all(tf.linalg.diag_part(P_pred) > tf.linalg.diag_part(P_old)))
        self.assertFalse(np.allclose(x_old.numpy(), self.ekf.x.numpy()))

    def test_update_step(self):
        """
        Test the updation step.
        """

        self.ekf.predict()

        # Simulate observations.
        z_sim = tf.zeros((self.cfg.meas_dim,), dtype=tf.float32)

        x_pred = self.ekf.x
        self.ekf.update(z_sim)

        # Check whether x is updated.
        self.assertFalse(np.allclose(x_pred.numpy(), self.ekf.x.numpy()))
        # Check whether P is symmetric.
        P_curr = self.ekf.P
        np.testing.assert_allclose(P_curr.numpy(), tf.transpose(P_curr).numpy(), atol=1e-5)


class TestAcousticPFPF(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        self.cfg = AcousticConfig()
        self.cfg.n_particles = 50
        self.model = AcousticModel(self.cfg)
        # Predicted covariance matrix by EKF
        self.P_pred = tf.eye(self.cfg.state_dim) * 0.1
        self.z_obs = tf.zeros((self.cfg.meas_dim,), dtype=tf.float32)

    def test_initialization(self):
        pf = AcousticPFPF(self.cfg, self.model, mode='LEDH')
        self.assertEqual(pf.particles.shape, (self.cfg.n_particles, self.cfg.state_dim))
        self.assertEqual(pf.weights.shape, (self.cfg.n_particles,))
        # Test whether the weights are normalized.
        self.assertAlmostEqual(tf.reduce_sum(pf.weights).numpy(), 1.0, places=5)

    def test_run_step_LEDH(self):
        """
        Test one-step LEDH.
        """

        pf = AcousticPFPF(self.cfg, self.model, mode='LEDH')

        try:
            x_est = pf.run_step(self.z_obs, self.P_pred)
        except Exception as e:
            self.fail(f"PFPF run_step (LEDH) failed with error: {e}")

        self.assertEqual(x_est.shape, (self.cfg.state_dim,))
        self.assertFalse(np.any(np.isnan(x_est.numpy())), "LEDH produced NaNs")

    def test_run_step_EDH(self):
        """
        Test one-step EDH.
        """

        pf = AcousticPFPF(self.cfg, self.model, mode='EDH')

        try:
            x_est = pf.run_step(self.z_obs, self.P_pred)
        except Exception as e:
            self.fail(f"PFPF run_step (EDH) failed with error: {e}")

        self.assertEqual(x_est.shape, (self.cfg.state_dim,))
        self.assertFalse(np.any(np.isnan(x_est.numpy())), "EDH produced NaNs")

if __name__ == '__main__':
    unittest.main()