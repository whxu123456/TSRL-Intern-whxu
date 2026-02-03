import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
import time
import tracemalloc


class Tracker:
    """
    Recording the running time and the memory use.
    """

    def __init__(self):
        self.start_time = 0
        self.peak_mem = 0

    def start(self):
        tracemalloc.start()
        self.start_time = time.time()

    def stop(self):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - self.start_time
        return elapsed, peak / 1024.0


def calculate_rmse(x_true, x_est):
    """
    calculate the root mean squared error (rmse)
    """

    if isinstance(x_true, tf.Tensor): x_true = x_true.numpy()
    if isinstance(x_est, tf.Tensor): x_est = x_est.numpy()
    return np.sqrt(np.mean((x_true - x_est) ** 2))


def calculate_omat(x_true, x_est, n_targets):
    """
    Calculate  optimal mass transfer (OMAT) (Li(17))
    x_true, x_est: [4*n_targets]
    """
    if isinstance(x_true, tf.Tensor): x_true = x_true.numpy()
    if isinstance(x_est, tf.Tensor): x_est = x_est.numpy()
    # get position [x, y]
    state_dim_per_target = len(x_true) // n_targets

    pos_true = x_true.reshape(n_targets, state_dim_per_target)[:, 0:2]
    pos_est = x_est.reshape(n_targets, state_dim_per_target)[:, 0:2]
    # cost matrix
    cost_matrix = np.zeros((n_targets, n_targets))
    for i in range(n_targets):
        for j in range(n_targets):
            cost_matrix[i, j] = np.linalg.norm(pos_true[i] - pos_est[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # average cost
    total_dist = cost_matrix[row_ind, col_ind].sum()
    return total_dist / n_targets


def is_positive_definite(matrix_list):
    """
    calculate the ratio of the positive definite covariance matrix
    """
    count = 0
    total = len(matrix_list)
    for mat in matrix_list:
        try:
            np.linalg.cholesky(mat)
            count += 1
        except np.linalg.LinAlgError:
            pass
    return count / total


def condition_number(cov_list):
    """
    calculate the condition number of the covariance matrix
    """

    conds = []
    for cov in cov_list:
        s = tf.linalg.svd(cov, compute_uv=False)
        c = (s[0] / (s[-1] + 1e-12)).numpy()
        conds.append(c)
    return np.array(conds)

def analyze_stability(cov_matrix):
    """
    Check whether the covariance matrix is positive definite
    """

    symmetry_error = np.linalg.norm(cov_matrix - np.transpose(cov_matrix, (0, 2, 1)), axis=(1, 2))
    return np.mean(symmetry_error), np.max(symmetry_error)