import pytest
import numpy as np
import tensorflow as tf
from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel
from src.filters.EKF_nonlinear_ssm import EKF_NonlinearSSM


class TestEKFNonlinear:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Initialize config and model
        self.cfg = NonlinearSSMConfig(T=100)
        self.model = NonlinearSSMModel(self.cfg)
        self.ekf = EKF_NonlinearSSM(self.cfg, self.model)
        # Fix random seed
        tf.random.set_seed(42)
        self.x_true, self.y_obs = self.model.generate_true_trajectory()

    def test_ekf_initialization(self):
        """Test EKF initialization"""
        assert self.ekf.x_est.shape == (self.cfg.state_dim,)
        assert self.ekf.P.shape == (self.cfg.state_dim, self.cfg.state_dim)
        assert self.ekf.x_pred.shape == (self.cfg.state_dim,)
        assert self.ekf.P_pred.shape == (self.cfg.state_dim, self.cfg.state_dim)

    def test_ekf_predict(self):
        """Test EKF prediction step"""
        t = 1
        P_pred = self.ekf.predict(t)
        assert P_pred.shape == (self.cfg.state_dim, self.cfg.state_dim)
        # Check that P_pred is positive definite
        assert np.all(np.linalg.eigvals(P_pred.numpy()) > 0)

    def test_ekf_update(self):
        """Test EKF update step"""
        # First predict to set x_pred and P_pred
        self.ekf.predict(t=1)
        # Then update
        z_obs = self.y_obs[1]
        self.ekf.update(z_obs)
        # Check that state and covariance are updated
        assert self.ekf.x_est.shape == (self.cfg.state_dim,)
        assert self.ekf.P.shape == (self.cfg.state_dim, self.cfg.state_dim)

    def test_ekf_run(self):
        """Test complete EKF run"""
        x_est_seq, run_time = self.ekf.run(self.y_obs)
        # Check output shapes
        assert x_est_seq.shape == (self.cfg.T, self.cfg.state_dim)
        assert run_time > 0.0
        # Threshold for EKF on this highly nonlinear model
        rmse = np.sqrt(np.mean((x_est_seq - self.x_true) ** 2))
        assert rmse < 25.0