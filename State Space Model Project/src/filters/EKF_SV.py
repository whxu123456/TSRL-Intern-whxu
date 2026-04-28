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