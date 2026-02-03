import unittest
import tensorflow as tf
import numpy as np
from src.models.linear_gaussian import LinearGaussianSSM
from src.filters.kalman import KalmanFilter


class TestLinearGaussianSSM(unittest.TestCase):
    def setUp(self):
        self.state_dim = 2
        self.process_noise_dim = 2
        self.obs_dim = 1
        self.obs_noise_dim = 1

        self.A = tf.eye(self.state_dim)
        self.B = tf.eye(self.process_noise_dim)
        self.C = tf.ones((self.obs_dim, self.state_dim))
        self.D = tf.eye(self.obs_noise_dim)
        self.Q = tf.eye(self.process_noise_dim)
        self.R = tf.eye(self.obs_noise_dim)
        self.initial_mean = tf.zeros(self.state_dim)
        self.initial_cov = tf.eye(self.state_dim)

        self.ssm = LinearGaussianSSM(
            self.state_dim, self.process_noise_dim, self.obs_dim, self.obs_noise_dim,
            self.A, self.B, self.C, self.D, self.Q, self.R,
            self.initial_mean, self.initial_cov
        )

    def test_initialization(self):
        """
        Test the initialization of covariance matrix.
        """

        expected_process_cov = self.B @ self.Q @ tf.transpose(self.B)
        expected_obs_cov = self.D @ self.R @ tf.transpose(self.D)

        np.testing.assert_array_almost_equal(self.ssm.process_noise_cov.numpy(), expected_process_cov.numpy())
        np.testing.assert_array_almost_equal(self.ssm.obs_noise_cov.numpy(), expected_obs_cov.numpy())

    def test_data_generation_shapes(self):
        """
        Test the shape of generated data.
        """

        num_timesteps = 10
        states, observations = self.ssm.generate_data(num_timesteps)
        self.assertEqual(states.shape, (num_timesteps + 1, self.state_dim)) # The "+1" is from x0
        self.assertEqual(observations.shape, (num_timesteps, self.obs_dim))


class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(42)

        self.state_dim = 2
        self.obs_dim = 1
        self.num_timesteps = 5

        # Test with a simplified system
        # x_k = x_{k-1}
        # y_k = x_k[0] + x_k[1]
        self.A = tf.eye(self.state_dim, dtype=tf.float32)
        self.C = tf.ones((self.obs_dim, self.state_dim), dtype=tf.float32)

        self.Q = tf.eye(self.state_dim, dtype=tf.float32) * 0.01
        self.R = tf.eye(self.obs_dim, dtype=tf.float32) * 0.01

        self.kf = KalmanFilter(
            self.state_dim, self.state_dim, self.obs_dim, self.obs_dim,
            self.num_timesteps, self.A, self.C, self.Q, self.R
        )

        self.initial_mean = tf.zeros(self.state_dim, dtype=tf.float32)
        self.initial_cov = tf.eye(self.state_dim, dtype=tf.float32)

    def test_predict_logic(self):
        """
        Test the prediction step
        """

        self.kf.initialize(self.initial_mean, self.initial_cov)
        # x_pred = A * x_prev = I * 0 = 0
        # P_pred = A * P * A^T + Q = I * I * I + 0.01*I = 1.01 * I
        pred_mean, pred_cov = self.kf.predict()
        expected_cov = self.initial_cov + self.Q
        np.testing.assert_array_almost_equal(pred_mean.numpy(), self.initial_mean.numpy())
        np.testing.assert_array_almost_equal(pred_cov.numpy(), expected_cov.numpy())

    def test_update_standard_vs_joseph(self):
        """
        Test whether Standard and Joseph is consistent
        when the covariance matrix updationis well conditioned.
        """

        observations = tf.random.normal((self.num_timesteps, self.obs_dim))
        # Run Standard
        self.kf.initialize(self.initial_mean, self.initial_cov)
        means_std, covs_std = self.kf.run_filter(observations, method='standard')
        # Run Joseph
        self.kf.initialize(self.initial_mean, self.initial_cov)
        means_joseph, covs_joseph = self.kf.run_filter(observations, method='joseph')

        np.testing.assert_allclose(means_std.numpy(), means_joseph.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(covs_std.numpy(), covs_joseph.numpy(), rtol=1e-5, atol=1e-6)

    def test_covariance_symmetry(self):
        """
        Test whether Joseph updation can guarantee the symmetric covariance matrix.
        """

        observations = tf.random.normal((self.num_timesteps, self.obs_dim))

        self.kf.initialize(self.initial_mean, self.initial_cov)
        _, covs_joseph = self.kf.run_filter(observations, method='joseph')

        for i in range(self.num_timesteps):
            cov = covs_joseph[i]
            diff = cov - tf.transpose(cov)
            max_diff = tf.reduce_max(tf.abs(diff))
            self.assertLess(max_diff.numpy(), 1e-6, f"Covariance at step {i} is not symmetric")

    def test_covariance_reduction(self):
        """
        Test whether (Posterior Cov < Prior Cov)
        """
        # Choose a small R
        self.kf.R = tf.eye(self.obs_dim) * 1e-8
        self.kf.initialize(self.initial_mean, self.initial_cov)
        obs = tf.constant([[1.0]], dtype=tf.float32)
        _, pred_cov = self.kf.predict()
        _, updated_cov = self.kf.update_standard(obs)
        det_pred = tf.linalg.det(pred_cov)
        det_upd = tf.linalg.det(updated_cov)

        self.assertLess(det_upd.numpy(), det_pred.numpy(), "Measurement update should reduce covariance determinant")

    def test_manual_calculation_1d(self):
        """
        Test with a 1-d system.
        """

        # 1-d System: x=1, A=1, C=1, Q=0, R=0 (Perfect model, Perfect observation)
        kf_1d = KalmanFilter(1, 1, 1, 1, 1,
                             tf.constant([[1.0]]), tf.constant([[1.0]]),
                             tf.constant([[0.0]]), tf.constant([[0.0]]))

        init_mean = tf.constant([1.0])
        init_cov = tf.constant([[1.0]])
        kf_1d.initialize(init_mean, init_cov)

        # Observation y = 2.0
        # Predict: x=1, P=1
        # Innovation: y - Cx = 2 - 1 = 1
        # S = CPC' + R = 1*1*1 + 0 = 1
        # K = PC'S^-1 = 1*1*1 = 1
        # Update x = x + K*innov = 1 + 1*1 = 2
        # Update P = (I - KC)P = (1 - 1*1)1 = 0

        obs = tf.constant([2.0])
        mean, cov = kf_1d.update_standard(obs)

        self.assertAlmostEqual(mean.numpy()[0], 2.0, places=5)
        self.assertAlmostEqual(cov.numpy()[0][0], 0.0, places=5)


if __name__ == '__main__':
    unittest.main()