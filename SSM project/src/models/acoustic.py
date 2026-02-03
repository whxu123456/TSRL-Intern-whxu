import tensorflow as tf
import numpy as np


class AcousticConfig:
    """
     Multi-Target Acoustic Tracking from Li (17).
     Parameters of Acoustic tracking model.
    """

    def __init__(self):
        self.T = 40 # time steps
        self.dt = 1
        self.n_targets = 4
        self.state_dim_per_target = 4 # [x, y, vx, vy]
        self.state_dim = 16 # 4*4
        self.meas_dim = 25 # # number of sensors
        self.n_particles = 500

        # data from sensorsXY.mat
        self.sensors_pos = tf.constant([
            [0, 0], [10, 0], [20, 0], [30, 0], [40, 0],
            [0, 10], [10, 10], [20, 10], [30, 10], [40, 10],
            [0, 20], [10, 20], [20, 20], [30, 20], [40, 20],
            [0, 30], [10, 30], [20, 30], [30, 30], [40, 30],
            [0, 40], [10, 40], [20, 40], [30, 40], [40, 40]
        ], dtype=tf.float32) # Shape: [25, 2]

        self.Amp = 10.0
        self.d0 = 0.1
        self.invPow = 1.0 # The power of norm in formula (38)
        self.sigma_w = np.sqrt(0.01) # observations standard deviation
        self.R_cov = 0.01*tf.eye(self.meas_dim) # observations covariance
        self.R_inv = tf.linalg.inv(self.R_cov)

        # covariance of the filtering process noise
        # Expand it to 16*16 matrix
        q_sub = np.array([[3, 0, 0.1, 0], [0, 3, 0, 0.1], [0.1, 0, 0.03, 0], [0, 0.1, 0, 0.03]], dtype=np.float32)
        self.Q_cov = tf.constant(np.kron(np.eye(self.n_targets), q_sub), dtype=tf.float32)
        self.Q_chol = tf.linalg.cholesky(self.Q_cov)
        self.Q_inv = tf.linalg.inv(self.Q_cov)

        # state transition matrix F
        f_sub = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.F = tf.constant(np.kron(np.eye(self.n_targets), f_sub), dtype=tf.float32)

        # initial target states
        self.x0_mean = tf.constant(
            [12, 6, 0.001, 0.001, 32, 32, -0.001, -0.005, 20, 13, -0.1, 0.01, 15, 35, 0.002, 0.002],
            dtype=tf.float32
        )
        self.P0 = tf.eye(self.state_dim) * 10.0


class AcousticModel:
    """
    Acoustic tracking model.
    """

    def __init__(self, config):
        self.cfg = config

    @tf.function
    def transition(self, x, noise=True):
        """
        Constant velocity model: trajectory of target states.
        input:
        x: [batch, state_dim]
        return:
        x_new: [batch, state_dim]
        """

        x_new = tf.matmul(x, self.cfg.F, transpose_b=True)
        if noise:
            noise_vec = tf.random.normal(tf.shape(x), dtype=tf.float32)
            x_new += tf.matmul(noise_vec, self.cfg.Q_chol, transpose_b=True)
        return x_new

    @tf.function
    def measurement(self, x):
        """
        Noiseless measurement from eq (38).
        input:
        x: [batch, state_dim]
        return:
        z_noiseless: [batch, meas_dim]
        """

        batch_size = tf.shape(x)[0]
        # reshape x: [batch, n_targets, 4]
        x_reshaped = tf.reshape(x, [batch_size, self.cfg.n_targets, 4])
        pos = x_reshaped[:, :, 0:2]
        # distance between every sensor and the targets
        # Sensors: [25, 2] -> [1, 1, 25, 2]
        # Pos: [batch, n_targets, 2] -> [batch, n_targets, 1, 2]
        sens_expanded = tf.reshape(self.cfg.sensors_pos, [1, 1, self.cfg.meas_dim, 2])
        pos_expanded = tf.expand_dims(pos, 2)
        dist_sq = tf.reduce_sum(tf.square(pos_expanded-sens_expanded), axis=3) # [batch, n_targets, 25]
        dist = tf.sqrt(dist_sq)
        # formula (38)
        signal = self.cfg.Amp/(tf.pow(dist, self.cfg.invPow)+self.cfg.d0) # [batch, n_targets, 25]
        z_noiseless = tf.reduce_sum(signal, axis=1) # [batch, 25]
        return z_noiseless

    @tf.function
    def measurement_jacobian(self, x):
        """
        Jacobian of measurement function (38).
        input:
        x: [batch, state_dim]
        return:
        H: [batch, meas_dim, state_dim]
        """

        batch_size = tf.shape(x)[0]
        # Reshape x: [batch, n_targets, 4]
        x_reshaped = tf.reshape(x, [batch_size, self.cfg.n_targets, 4])
        pos = x_reshaped[:, :, 0:2]
        # distance between every sensor and the targets
        # Sensors: [25, 2] -> [1, 1, 25, 2]
        # Pos: [batch, n_targets, 2] -> [batch, n_targets, 1, 2]
        sens_expanded = tf.reshape(self.cfg.sensors_pos, [1, 1, self.cfg.meas_dim, 2])
        pos_expanded = tf.expand_dims(pos, 2)
        # diff: [batch, n_targets, 25, 2] (x - s)
        diff = pos_expanded - sens_expanded
        dist_sq = tf.reduce_sum(tf.square(diff), axis=3)
        dist = tf.sqrt(dist_sq)
        # d(Amp/(r+d0))/dx = -Amp/(r+d0)^2 * (x-s)/r
        denom = (dist+self.cfg.d0)**2*dist # [batch, n_targets, 25]
        factor = -self.cfg.Amp/denom # [batch, n_targets, 25]
        factor = tf.expand_dims(factor, -1) # [batch, n_targets, 25, 1]
        # dH_dpos: [batch, n_targets, 25, 2]
        dH_dpos = factor*diff
        # Jacobian [batch, 25, 16]
        H_cols = []
        for c in range(self.cfg.n_targets):
            # derivative of c-th target
            dx = dH_dpos[:, c, :, 0] # [batch, 25]
            dy = dH_dpos[:, c, :, 1] # [batch, 25]
            zeros = tf.zeros_like(dx)
            H_cols.extend([dx, dy, zeros, zeros])
        H = tf.stack(H_cols, axis=2) # [batch, 25, 16]
        return H

    def limit_boundary(self, x, min_val=-5, max_val=45):
        """
        Limit the trajectory in the 40*40 square (Similar to Fig 1 in Li(17))
        input: x: [state_dim]
        return: x_np: [state_dim]
        """

        x_np = x.numpy()
        n_targets = 4
        for i in range(n_targets):
            px_idx = i * 4
            py_idx = i * 4 + 1
            vx_idx = i * 4 + 2
            vy_idx = i * 4 + 3

            # Check X
            if x_np[px_idx] < min_val:
                x_np[px_idx] = min_val + (min_val - x_np[px_idx])
                x_np[vx_idx] *= -1
            elif x_np[px_idx] > max_val:
                x_np[px_idx] = max_val - (x_np[px_idx] - max_val)
                x_np[vx_idx] *= -1

            # Check Y
            if x_np[py_idx] < min_val:
                x_np[py_idx] = min_val + (min_val - x_np[py_idx])
                x_np[vy_idx] *= -1
            elif x_np[py_idx] > max_val:
                x_np[py_idx] = max_val - (x_np[py_idx] - max_val)
                x_np[vy_idx] *= -1

        return tf.constant(x_np, dtype=tf.float32)