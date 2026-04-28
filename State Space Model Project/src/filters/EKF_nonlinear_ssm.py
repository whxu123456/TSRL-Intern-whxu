import numpy as np
import tensorflow as tf
import time
from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel


class EKF_NonlinearSSM:
    """
    Extended Kalman Filter for Section 3.1 nonlinear model
    Used for predicting covariance matrix P in Li (2017) PF-PF
    """

    def __init__(self, config: NonlinearSSMConfig, model: NonlinearSSMModel):
        self.cfg = config
        self.model = model
        # Initialize state and covariance
        self.x_est = tf.identity(self.cfg.x0_mean)
        self.P = tf.identity(self.cfg.P0)
        self.x_pred = tf.identity(self.x_est)
        self.P_pred = tf.identity(self.P)

    def predict(self, t):
        """
        EKF prediction step, compute prior mean and covariance
        Args:
            t: Current time step
        Returns:
            P_pred: Predicted covariance matrix [state_dim, state_dim]
        """
        # 1. State mean prediction
        x_pred = self.model.transition_mean(self.x_est, t)

        # 2. State transition Jacobian matrix F = df/dx
        # Manually derived transition function Jacobian (1D scalar)
        x_scalar = self.x_est[0]
        df_dx = self.cfg.alpha + self.cfg.beta * (1.0 - x_scalar**2) / (1.0 + x_scalar**2)**2
        # Reshape scalar tensor to 1x1 matrix directly (avoid tf.constant with tensor input)
        F = tf.reshape(df_dx, [self.cfg.state_dim, self.cfg.state_dim])

        # 3. Covariance prediction: P_pred = F @ P @ F^T + Q
        P_pred = F @ self.P @ tf.transpose(F) + self.cfg.Q_cov

        # Save predicted values for update step
        self.x_pred = x_pred
        self.P_pred = P_pred
        return P_pred

    def update(self, z_obs):
        """
        EKF update step, correct state and covariance
        Args:
            z_obs: Current observation [meas_dim]
        """
        # 1. Measurement Jacobian matrix
        H = self.model.measurement_jacobian(self.x_pred)
        # Ensure H is always shape [meas_dim, state_dim] (2D matrix)
        H = tf.reshape(H, [self.cfg.meas_dim, self.cfg.state_dim])

        # 2. Measurement residual
        z_pred = self.model.measurement_model(self.x_pred)
        y = tf.reshape(z_obs, [-1]) - tf.reshape(z_pred, [-1])

        # 3. Covariance update
        S = H @ self.P_pred @ tf.transpose(H) + self.cfg.R_cov
        K = self.P_pred @ tf.transpose(H) @ tf.linalg.inv(S + 1e-12 * tf.eye(self.cfg.meas_dim))

        # State update
        self.x_est = self.x_pred + tf.matmul(y[None, :], K, transpose_b=True)[0]

        # Joseph form covariance update (more numerically stable)
        I = tf.eye(self.cfg.state_dim)
        I_KH = I - K @ H
        term1 = I_KH @ self.P_pred @ tf.transpose(I_KH)
        term2 = K @ self.cfg.R_cov @ tf.transpose(K)
        self.P = term1 + term2

    def run(self, y_obs):
        """
        Run complete EKF filtering
        Args:
            y_obs: Observation sequence [T, meas_dim]
        Returns:
            x_est_seq: State estimation sequence [T, state_dim]
            run_time: Runtime
        """
        T = y_obs.shape[0]
        x_est_seq = np.zeros((T, self.cfg.state_dim), dtype=np.float32)
        start_time = time.time()

        self.update(y_obs[0])
        x_est_seq[0] = self.x_est.numpy()

        # Recursive filtering
        for t in range(1, T):
            self.predict(t)
            self.update(y_obs[t])
            x_est_seq[t] = self.x_est.numpy()

        run_time = time.time() - start_time
        return x_est_seq, run_time