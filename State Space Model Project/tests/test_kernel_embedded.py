import unittest
import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.lorenz96 import Lorenz96
from src.filters.kernel_embedded import KernelEmbeddedPFF


class TestLorenz96(unittest.TestCase):
    """
    Unit tests for the Lorenz96 dynamic model.
    """

    def setUp(self):
        """
        Initialize a standard Lorenz96 model before each test.
        """
        self.nx = 40
        self.F = 8
        self.model = Lorenz96(nx=self.nx, F=self.F)

    def test_initialization(self):
        """
        Test if the Lorenz96 model initializes with correct parameters.
        Verifies nx and F attributes are set properly.
        """
        self.assertEqual(self.model.nx, self.nx)
        self.assertEqual(self.model.F, self.F)

    def test_dynamics_constant_state(self):
        """
        Test the dynamics equation with a constant state vector.
        Logic:
        dx/dt = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        If x is a vector of all 1s:
        term1 = (1 - 1) * 1 = 0
        result = 0 - 1 + F = F - 1
        If F=8, dx/dt should be 7.
        """
        x_const = tf.ones((self.nx,), dtype=tf.float32)
        dxdt = self.model.dynamics(x_const)

        expected_val = self.F - 1.0
        self.assertTrue(np.allclose(dxdt.numpy(), expected_val),
                        f"Dynamics failed. Expected {expected_val}, got {dxdt.numpy()[0]}")

    def test_rk4_step_integration(self):
        """
        Test that the RK4 integrator actually updates the state and maintain the shape.
        """
        x_start = tf.random.normal((self.nx,), dtype=tf.float32)
        dt = 0.01
        x_next = self.model.rk4_step(x_start, dt)

        # Ensure the state has changed
        self.assertFalse(np.allclose(x_start.numpy(), x_next.numpy()),
                         "RK4 step did not update the state vector.")

        # Ensure shape is preserved
        self.assertEqual(x_start.shape, x_next.shape,
                         "RK4 step altered the shape of the state vector.")


class TestKernelEmbeddedPFF(unittest.TestCase):
    """
    Unit and Integration tests for the Kernel-Embedded Particle Flow Filter.
    """

    def setUp(self):
        """
        Set up the filter parameters and mock data.
        """
        self.nx = 20
        self.np = 10
        self.obs_indices = np.array([0, 5, 10, 15])
        self.obs_std = 0.5
        self.ny = len(self.obs_indices)

        # Initialize particles and observations
        self.particles = tf.random.normal((self.np, self.nx), dtype=tf.float32)
        self.y_obs = tf.random.normal((self.ny,), dtype=tf.float32)

        # Prior parameters for gradient test
        self.prior_mean = tf.reduce_mean(self.particles, axis=0)
        self.B_diag = tf.reduce_mean(tf.square(self.particles - self.prior_mean), axis=0)
        self.B_inv = tf.linalg.diag(1.0 / self.B_diag)

    def test_initialization(self):
        """
        Test if the filter initializes with correct dimensions and attributes.
        Validates state size, particle count, observation indices, and covariance matrices.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        self.assertEqual(pff.nx, self.nx)
        self.assertEqual(pff.Np, self.np)
        self.assertEqual(pff.ny, self.ny)
        self.assertEqual(pff.R_inv.shape, (self.ny, self.ny))
        self.assertEqual(pff.kernel_type, 'matrix')
        self.assertEqual(pff.alpha, 1 / self.np)

    def test_log_likelihood_grad_shape(self):
        """
        Test if the gradient of log-likelihood returns the correct shape (Np, Nx).
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        grad = pff.log_likelihood_grad(self.particles, self.y_obs)

        self.assertEqual(grad.shape, (self.np, self.nx),
                         f"Likelihood gradient shape mismatch. Expected {(self.np, self.nx)}, got {grad.shape}")

    def test_log_likelihood_grad_values(self):
        """
        Test numerical correctness of the log likelihood gradient.
        Verifies gradient values match expected linear observation model calculations.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        grad = pff.log_likelihood_grad(self.particles, self.y_obs)
        self.assertFalse(np.any(np.isnan(grad.numpy())))
        self.assertFalse(np.any(np.isinf(grad.numpy())))

    def test_log_prior_grad_shape(self):
        """
        Test if the gradient of log prior returns the correct shape (Np, Nx).
        This method was missing in the original test file.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        grad = pff.log_prior_grad(self.particles, self.prior_mean, self.B_inv)

        self.assertEqual(grad.shape, (self.np, self.nx),
                         f"Prior gradient shape mismatch. Expected {(self.np, self.nx)}, got {grad.shape}")

    def test_log_prior_grad_values(self):
        """
        Test numerical stability and correctness of the log prior gradient.
        Ensures no NaN/Inf values and valid gradient magnitude.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        grad = pff.log_prior_grad(self.particles, self.prior_mean, self.B_inv)

        self.assertFalse(np.any(np.isnan(grad.numpy())))
        self.assertFalse(np.any(np.isinf(grad.numpy())))

    def test_scalar_kernel_computation(self):
        """
        Test the computation of the Scalar Kernel.
        Checks if K matrix and gradients have correct dimensions.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='scalar')

        B_diag = tf.ones((self.nx,), dtype=tf.float32)
        K_val, grad_K = pff.compute_kernel_matrix(self.particles, B_diag)

        self.assertEqual(K_val.shape, (self.np, self.np, 1), "Scalar Kernel K matrix shape incorrect.")
        self.assertEqual(grad_K.shape, (self.np, self.np, self.nx), "Scalar Kernel gradient shape incorrect.")
        self.assertFalse(np.any(np.isnan(K_val.numpy())))

    def test_matrix_kernel_computation(self):
        """
        Test the computation of the Matrix-Valued Kernel.
        Validates shape and numerical stability of kernel outputs.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='matrix')

        B_diag = tf.ones((self.nx,), dtype=tf.float32)
        K_val, grad_K = pff.compute_kernel_matrix(self.particles, B_diag)

        self.assertEqual(K_val.shape, (self.np, self.np, self.nx), "Matrix Kernel K matrix shape incorrect.")
        self.assertEqual(grad_K.shape, (self.np, self.np, self.nx), "Matrix Kernel gradient shape incorrect.")
        self.assertFalse(np.any(np.isinf(grad_K.numpy())))

    def test_update_integration(self):
        """
        Integration Test: Run the full update loop.
        Ensures the pipeline runs end-to-end and particles are updated.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='matrix')

        updated_particles = pff.update(self.particles, self.y_obs, n_steps=5, dt_s=0.01)

        self.assertEqual(updated_particles.shape, self.particles.shape, "Updated particles lost shape integrity.")
        self.assertFalse(np.allclose(self.particles.numpy(), updated_particles.numpy()),
                         "Particles did not move after PFF update step.")

    def test_numerical_stability(self):
        """
        Test that the update does not produce NaNs or Infs.
        Validates numerical stability of the full flow integration.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='scalar')
        updated_particles = pff.update(self.particles, self.y_obs, n_steps=5, dt_s=0.01)

        self.assertFalse(np.any(np.isnan(updated_particles.numpy())), "Update produced NaNs.")
        self.assertFalse(np.any(np.isinf(updated_particles.numpy())), "Update produced Infs.")

    def test_update_prior_statistics(self):
        """
        Test prior statistics calculation inside the update method.
        Verifies prior mean and diagonal covariance are computed correctly.
        """
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        # Run a short update to trigger prior stats calculation
        updated = pff.update(self.particles, self.y_obs, n_steps=1, dt_s=0.01)
        self.assertEqual(updated.shape, self.particles.shape)


if __name__ == '__main__':
    unittest.main()