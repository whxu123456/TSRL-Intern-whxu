import unittest
import numpy as np
import os
import shutil
import tempfile
import sys
from unittest.mock import patch
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.utils import plot
from src.utils.plot import (
    save_or_show,
    plot_trajectory_comparison,
    plot_error_metrics,
    plot_metrics_bar,
    plot_acoustic_tracking,
    plot_lorenz_collapse
)


class TestPlot(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        np.random.seed(123)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_save_or_show_display(self, mock_show):
        fig = plt.figure()
        save_or_show(fig, filename=None)
        mock_show.assert_called_once()

    def test_save_or_show_save(self):
        fig = plt.figure()
        filename = "test_fig.png"
        save_or_show(fig, filename=filename, output_dir=self.test_dir)
        expected_path = os.path.join(self.test_dir, filename)
        self.assertTrue(os.path.exists(expected_path))

    @patch.object(plot, 'save_or_show')
    def test_plot_trajectory_comparison(self, mock_save_show):
        T = 50
        x_true = np.random.randn(T)
        estimates = {'EKF': np.random.randn(T)}
        plot_trajectory_comparison(x_true, estimates, filename="traj.png")
        self.assertTrue(mock_save_show.called)
        args, kwargs = mock_save_show.call_args
        actual_filename = kwargs.get('filename')

        if actual_filename is None and len(args) >= 2:
            actual_filename = args[1]

        self.assertEqual(actual_filename, "traj.png",
                         f"Cannot find the expected file name. Args: {args}, Kwargs: {kwargs}")

    @patch.object(plot, 'save_or_show')
    def test_plot_error_metrics(self, mock_save_show):
        errors = {'EKF': [0.1, 0.05]}
        plot_error_metrics(errors)
        self.assertTrue(mock_save_show.called)

    @patch.object(plot, 'save_or_show')
    def test_plot_metrics_bar(self, mock_save_show):
        metrics = {'EKF': 0.5}
        plot_metrics_bar(metrics)
        self.assertTrue(mock_save_show.called)

    @patch.object(plot, 'save_or_show')
    def test_plot_acoustic_tracking(self, mock_save_show):
        n_targets = 1
        T = 10
        x_true = np.random.randn(T, 4 * n_targets)
        sensors = np.array([[0, 0]])
        est_dict = {'Proposed': np.random.randn(T, 4 * n_targets)}

        plot_acoustic_tracking(sensors, x_true, est_dict, n_targets)
        self.assertTrue(mock_save_show.called)

    @patch.object(plot, 'save_or_show')
    def test_plot_lorenz_collapse(self, mock_save_show):
        N = 10
        dim = 40
        prior = np.random.randn(N, dim)
        posterior = np.random.randn(N, dim)
        truth = np.random.randn(dim)
        obs_indices = [0, 2, 4]

        plot_lorenz_collapse(prior, posterior, truth, obs_indices)
        self.assertTrue(mock_save_show.called)


if __name__ == '__main__':
    unittest.main()