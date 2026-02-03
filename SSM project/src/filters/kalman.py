import tensorflow as tf

class KalmanFilter:
    # The recursive process of Kalman Filter
    def __init__(self, state_dim, process_noise_dim, obs_dim, obs_noise_dim, num_timesteps, A, C, Q, R):
        self.state_dim = state_dim  # n_x
        self.process_noise_dim = process_noise_dim  # n_v
        self.obs_dim = obs_dim  # n_y
        self.obs_noise_dim = obs_noise_dim  # n_w
        self.num_timesteps = num_timesteps
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R

    def initialize(self, initial_mean, initial_cov):
        self.state_mean = initial_mean
        self.state_cov = initial_cov
        self.predicted_means = []
        self.predicted_covs = []
        self.filtered_means = []
        self.filtered_covs = []

    def predict(self):
        # prediction
        # x_{n|n-1} = A x_{n-1|n-1}
        predicted_mean = tf.linalg.matvec(self.A, self.state_mean)

        # P_{n|n-1} = A P_{n-1|n-1} A^T + Q
        predicted_cov = self.A @ self.state_cov @ tf.transpose(self.A) + self.Q

        self.predicted_means.append(predicted_mean)
        self.predicted_covs.append(predicted_cov)

        return predicted_mean, predicted_cov

    def update_standard(self, observation):
        # updation
        pred_mean, pred_cov = self.predict()

        # y_n - C x_{n|n-1}
        y_delta = observation - tf.linalg.matvec(self.C, pred_mean)

        # S_n = C P_{n|n-1} C^T + R
        S_n = self.C @ pred_cov @ tf.transpose(self.C) + self.R

        # K_n = P_{n|n-1} C^T S_n^{-1}
        kalman_gain = pred_cov @ tf.transpose(self.C) @ tf.linalg.inv(S_n)

        # x_{n|n} = x_{n|n-1} + K_n (y_n - C x_{n|n-1})
        updated_mean = pred_mean + tf.linalg.matvec(kalman_gain, y_delta)

        # P_{n|n} = (I - K_n C) P_{n|n-1}
        I = tf.eye(self.state_dim, dtype=tf.float32)
        updated_cov = (I - kalman_gain @ self.C) @ pred_cov

        self.state_mean = updated_mean
        self.state_cov = updated_cov
        self.filtered_means.append(updated_mean)
        self.filtered_covs.append(updated_cov)

        return updated_mean, updated_cov

    def update_joseph(self, observation):
        # update by joseph
        pred_mean, pred_cov = self.predict()

        # y_n - C x_{n|n-1}
        y_delta = observation - tf.linalg.matvec(self.C, pred_mean)

        # S_n = C P_{n|n-1} C^T + R
        S_n = self.C @ pred_cov @ tf.transpose(self.C) + self.R

        # K_n = P_{n|n-1} C^T S_n^{-1}
        kalman_gain = pred_cov @ tf.transpose(self.C) @ tf.linalg.inv(S_n)

        # x_{n|n} = x_{n|n-1} + K_n (y_n - C x_{n|n-1})
        updated_mean = pred_mean + tf.linalg.matvec(kalman_gain, y_delta)

        # Joseph update
        I = tf.eye(self.state_dim, dtype=tf.float32)
        temp = I - kalman_gain @ self.C
        updated_cov = (temp @ pred_cov @ tf.transpose(temp) +
                       kalman_gain @ self.R @ tf.transpose(kalman_gain))

        self.state_mean = updated_mean
        self.state_cov = updated_cov
        self.filtered_means.append(updated_mean)
        self.filtered_covs.append(updated_cov)

        return updated_mean, updated_cov

    def run_filter(self, observations, method='standard'):
        if method == 'standard':
            for t in range(self.num_timesteps):
                self.update_standard(observations[t])
            return (tf.stack(self.filtered_means), tf.stack(self.filtered_covs))
        elif method == 'joseph':
            for t in range(self.num_timesteps):
                self.update_joseph(observations[t])
            return (tf.stack(self.filtered_means), tf.stack(self.filtered_covs))