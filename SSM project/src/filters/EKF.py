import numpy as np
import tensorflow as tf
import time


class EKF_SV:
    """
    Extended Kalman filter.
    Take the logarithm of Y_n^2, separate noise from X_n.
    Z_n=log(Y_n^2)+X_n+log(W_n^2)
    """

    def __init__(self, model):
        self.model = model
        self.noise_mean = -1.27
        self.noise_var = np.pi**2/2

    def run(self, y_obs):
        T = y_obs.shape[0]
        z_obs = tf.math.log(y_obs**2 + 1e-8)
        x_est = np.zeros(T)
        x_curr = 0.0
        P_curr = self.model.sigma**2/(1 - self.model.alpha**2)

        start_time = time.time()
        for t in range(T):
            x_pred = self.model.alpha*x_curr
            P_pred = self.model.alpha**2*P_curr + self.model.sigma**2
            H = 1
            z_pred = np.log(self.model.beta**2) + x_pred + self.noise_mean
            z_delta = z_obs[t] - z_pred
            S = H*P_pred*H + self.noise_var
            K = P_pred*H/S
            x_curr = x_pred + K*z_delta
            P_curr = (1 - K*H)*P_pred
            x_est[t] = x_curr
        return x_est, time.time() - start_time


class EKF_Predictor:
    """
    Used for Acoustic Tracking PFPF (Li(17)).
    EKF prediction to estimate predictive covriance matrix P_{k|k-1}.
    """

    def __init__(self, config, model):
        self.cfg = config
        self.model = model
        self.P = self.cfg.P0 # initial covariance matrix
        self.x = tf.reshape(self.cfg.x0_mean, [1, -1]) # initial mean

    def predict(self):
        # x_k|k-1 = F x_k-1|k-1
        self.x = tf.matmul(self.x, self.cfg.F, transpose_b=True)
        # P_k|k-1 = F P_k-1|k-1 F^T + Q
        self.P = self.cfg.F @ self.P @ tf.transpose(self.cfg.F) + self.cfg.Q_cov
        return self.P

    def update(self, z):
        H = self.model.measurement_jacobian(self.x)
        H = tf.squeeze(H, 0)
        S = H @ self.P @ tf.transpose(H) + self.cfg.R_cov
        K = self.P @ tf.transpose(H) @ tf.linalg.inv(S)
        z_pred = self.model.measurement(self.x)
        y = tf.reshape(z, [1, -1]) - z_pred
        self.x = self.x + tf.matmul(y, K, transpose_b=True)
        I = tf.eye(self.cfg.state_dim)
        # self.P = (I - K @ H) @ self.P (standard update)
        # Joseph update
        I_KH = I - K @ H
        term1 = I_KH @ self.P @ tf.transpose(I_KH)
        term2 = K @ self.cfg.R_cov @ tf.transpose(K)
        self.P = term1 + term2


class EKF_SV_Raw:
    """
    Raw EKF: Tries to filter without Log-transform.
    Observation: z = y^2 = beta^2 * exp(x) * epsilon^2
    Use y^2 because E[y]=0, which kills the gradient.
    """

    def __init__(self, model):
        self.model = model

    def run(self, y_obs):
        T = y_obs.shape[0]
        # Observation is y^2 directly
        z_obs = y_obs ** 2

        x_est = np.zeros(T)
        x_curr = 0.0
        P_curr = self.model.sigma ** 2 / (1 - self.model.alpha ** 2)

        start_time = time.time()
        for t in range(T):
            # 1. Predict
            x_pred = self.model.alpha * x_curr
            P_pred = self.model.alpha ** 2 * P_curr + self.model.sigma ** 2

            # 2. Update
            # h(x) = E[y^2|x] = beta^2 * exp(x) * E[eps^2] = beta^2 * exp(x)
            h_x = (self.model.beta ** 2) * np.exp(x_pred)

            # Jacobian H = d(h_x)/dx = beta^2 * exp(x) = h_x
            H = h_x

            # Effective Noise Variance R
            # Var(y^2) = Var(beta^2 * exp(x) * eps^2)
            #          = (beta^2 * exp(x))^2 * Var(eps^2)
            #          = h_x^2 * 2  (Since eps^2 is Chi-sq(1), Var=2)
            R_eff = (h_x ** 2) * 2.0 + 1e-6

            S = H * P_pred * H + R_eff
            K = P_pred * H / S

            # Innovation
            y_res = z_obs[t] - h_x

            x_curr = x_pred + K * y_res
            P_curr = (1 - K * H) * P_pred

            x_est[t] = x_curr

        return x_est, time.time() - start_time