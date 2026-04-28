import tensorflow as tf


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
        z_pred = self.model.measurement_model(self.x)
        y = tf.reshape(z, [1, -1]) - z_pred
        self.x = self.x + tf.matmul(y, K, transpose_b=True)
        I = tf.eye(self.cfg.state_dim)
        # self.P = (I - K @ H) @ self.P (standard update)
        # Joseph update
        I_KH = I - K @ H
        term1 = I_KH @ self.P @ tf.transpose(I_KH)
        term2 = K @ self.cfg.R_cov @ tf.transpose(K)
        self.P = term1 + term2

