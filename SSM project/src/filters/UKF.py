import numpy as np
import tensorflow as tf
import time


class UKF_SV:
    """
    Unsent Kalman Filter.
    Take the logarithm of Y_n^2, separate noise from X_n.
    Z_n=log(Y_n^2)+X_n+log(W_n^2)
    """
    def __init__(self, model):
        self.model = model
        self.noise_mean = -1.27
        self.noise_var = np.pi ** 2 / 2
        self.kappa = 2

    def run(self, y_obs):
        T = y_obs.shape[0]
        z_obs = tf.math.log(y_obs ** 2 + 1e-8)
        x_est = np.zeros(T)
        x_curr = 0
        P_curr = self.model.sigma ** 2 / (1 - self.model.alpha ** 2)

        start_time = time.time()
        for t in range(T):
            sigma_points = [x_curr, x_curr + np.sqrt((1 + self.kappa) * P_curr),
                            x_curr - np.sqrt((1 + self.kappa) * P_curr)]
            weights = [self.kappa / (1 + self.kappa), 0.5 / (1 + self.kappa), 0.5 / (1 + self.kappa)]

            x_sig_pred = [self.model.alpha * s for s in sigma_points]
            x_pred = sum([w * s for w, s in zip(weights, x_sig_pred)])
            P_pred = sum([w * (s - x_pred) ** 2 for w, s in zip(weights, x_sig_pred)]) + self.model.sigma ** 2

            sigma_points_pred = [x_pred, x_pred + np.sqrt((1 + self.kappa) * P_pred),
                                 x_pred - np.sqrt((1 + self.kappa) * P_pred)]
            z_sig_pred = [np.log(self.model.beta ** 2) + s + self.noise_mean for s in sigma_points_pred]
            z_pred = sum([w * z for w, z in zip(weights, z_sig_pred)])
            S = sum([w * (z - z_pred) ** 2 for w, z in zip(weights, z_sig_pred)]) + self.noise_var

            Pxz = sum([w * (s - x_pred) * (z - z_pred) for w, s, z in zip(weights, sigma_points_pred, z_sig_pred)])
            K = Pxz / S
            x_curr = x_pred + K * (z_obs[t] - z_pred)
            P_curr = P_pred - K * S * K
            x_est[t] = x_curr

        return x_est, time.time() - start_time


class UKF_SV_Raw:
    """
    Raw UKF: Uses raw y^2 without log-transform.
    Demonstrates failure due to non-Gaussian multiplicative noise.
    """

    def __init__(self, model):
        self.model = model
        self.kappa = 2

    def run(self, y_obs):
        T = y_obs.shape[0]
        z_obs = y_obs ** 2  # Use raw squared observations because E[y]=0, which kills the gradient.

        x_est = np.zeros(T)
        x_curr = 0.0
        P_curr = self.model.sigma ** 2 / (1 - self.model.alpha ** 2)

        start_time = time.time()
        for t in range(T):
            # Sigma Points
            sigma_points = [x_curr,
                            x_curr + np.sqrt((1 + self.kappa) * P_curr),
                            x_curr - np.sqrt((1 + self.kappa) * P_curr)]
            weights = [self.kappa / (1 + self.kappa), 0.5 / (1 + self.kappa), 0.5 / (1 + self.kappa)]

            # Time Update
            x_sig_pred = [self.model.alpha * s for s in sigma_points]
            x_pred = sum([w * s for w, s in zip(weights, x_sig_pred)])
            P_pred = sum([w * (s - x_pred) ** 2 for w, s in zip(weights, x_sig_pred)]) + self.model.sigma ** 2

            # Measurement Update (Non-linear raw space)
            # z = beta^2 * exp(x) * eps^2
            # Approximate E[z] using sigma points passed through h(x) = beta^2*exp(x)
            sigma_points_pred = [x_pred,
                                 x_pred + np.sqrt((1 + self.kappa) * P_pred),
                                 x_pred - np.sqrt((1 + self.kappa) * P_pred)]

            # Expected observation given state (assume E[eps^2]=1)
            z_sig_pred = [(self.model.beta ** 2) * np.exp(s) for s in sigma_points_pred]

            z_pred = sum([w * z for w, z in zip(weights, z_sig_pred)])

            # Approximate R (Observation Noise Variance)
            # R is highly state dependent: 2 * (beta^2 * exp(x))^2
            # I use the predicted x to estimate current noise level
            R_adaptive = 2.0 * (z_pred ** 2) + 1e-6

            S = sum([w * (z - z_pred) ** 2 for w, z in zip(weights, z_sig_pred)]) + R_adaptive

            Pxz = sum([w * (s - x_pred) * (z - z_pred) for w, s, z in zip(weights, sigma_points_pred, z_sig_pred)])

            K = Pxz / S
            x_curr = x_pred + K * (z_obs[t] - z_pred)
            P_curr = P_pred - K * S * K
            x_est[t] = x_curr

        return x_est, time.time() - start_time