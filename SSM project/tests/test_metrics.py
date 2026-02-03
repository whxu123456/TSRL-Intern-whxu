import unittest
import numpy as np
import tensorflow as tf
import time
from src.utils.metrics import Tracker, calculate_rmse, calculate_omat, is_positive_definite, condition_number


class TestMetrics(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_tracker(self):
        """
        Test Tracker for time and memory
        """

        tracker = Tracker()
        tracker.start()
        # Simulation
        _ = [i for i in range(10000)]
        time.sleep(0.01)

        elapsed, peak_mem = tracker.stop()

        self.assertGreater(elapsed, 0)
        self.assertGreaterEqual(peak_mem, 0)
        self.assertIsInstance(elapsed, float)

    def test_calculate_rmse(self):
        """Test RMSE calculation (Numpy and Tensor)"""
        # Case 1: Numpy input
        x_true = np.array([1.0, 2.0, 3.0])
        x_est = np.array([1.0, 2.0, 4.0])  # error: 0, 0, 1 -> MSE=1/3 -> RMSE=sqrt(1/3)

        rmse = calculate_rmse(x_true, x_est)
        self.assertAlmostEqual(rmse, np.sqrt(1 / 3), places=5)

        # Case 2: Tensor input
        x_true_tf = tf.constant([1.0, 1.0])
        x_est_tf = tf.constant([3.0, 3.0])  # error 2, 2 -> MSE=4 -> RMSE=2
        rmse_tf = calculate_rmse(x_true_tf, x_est_tf)
        self.assertAlmostEqual(rmse_tf, 2.0, places=5)

    def test_calculate_omat(self):
        """
        Test OMAT (Optimal Mass Transfer).
        Kuhn-Munkres Algorithm.
        """
        n_targets = 2
        # The dimension of every target is 4 (x, y, vx, vy),
        # OMAT is calculated by the first two dimensions (x, y).

        # Target 1: (0,0), Target 2: (10,10)
        # [x1, y1, ..., x2, y2, ...]
        x_true = np.array([0, 0, 0, 0, 10, 10, 0, 0])

        # (0,0)->(0,1), (10,10)->(10,11)
        # 顺序与 True 一致
        x_est_ordered = np.array([0, 1, 0, 0, 10, 11, 0, 0])

        dist_ordered = calculate_omat(x_true, x_est_ordered, n_targets)
        # average distance = (1 + 1) / 2 = 1.0
        self.assertAlmostEqual(dist_ordered, 1.0, places=5)
        x_est_swapped = np.array([10, 11, 0, 0, 0, 1, 0, 0])

        dist_swapped = calculate_omat(x_true, x_est_swapped, n_targets)
        self.assertAlmostEqual(dist_swapped, 1.0, places=5)

    def test_is_positive_definite(self):
        """
        Test the positive definiteness of the metrics.
        """

        pd_mat = np.eye(3)
        non_pd_mat = np.zeros((3, 3))

        ratio = is_positive_definite([pd_mat, non_pd_mat, pd_mat])
        # two of the three matrix are pd -> 0.666...
        self.assertAlmostEqual(ratio, 2 / 3, places=5)

    def test_condition_number(self):
        """
        Test the calculation of condition number
        """
        mat1 = tf.eye(2, dtype=tf.float32)
        mat2 = tf.constant([[1.0, 1.0], [2.0, 2.0]], dtype=tf.float32)

        conds = condition_number([mat1, mat2])

        self.assertAlmostEqual(conds[0], 1.0, places=4)
        self.assertGreater(conds[1], 1e5)


if __name__ == '__main__':
    unittest.main()