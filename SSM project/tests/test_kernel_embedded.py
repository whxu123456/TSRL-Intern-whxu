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
        self.np = 10  # Number of particles
        self.obs_indices = np.array([0, 5, 10, 15])
        self.obs_std = 0.5

        # Initialize particles and observations
        self.particles = tf.random.normal((self.np, self.nx), dtype=tf.float32)
        self.y_obs = tf.random.normal((len(self.obs_indices),), dtype=tf.float32)

    def test_initialization(self):
        """
        Test if the filter initializes with correct dimensions.
        """

        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        self.assertEqual(pff.R_inv.shape, (4, 4), "Inverse covariance matrix R_inv has wrong shape.")

    def test_log_likelihood_grad_shape(self):
        """
        Test if the gradient of log-likelihood returns the correct shape (Np, Nx).
        """

        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std)
        grad = pff.log_likelihood_grad(self.particles, self.y_obs)

        self.assertEqual(grad.shape, (self.np, self.nx),
                         f"Likelihood gradient shape mismatch. Expected {(self.np, self.nx)}, got {grad.shape}")

    def test_scalar_kernel_computation(self):
        """
        Test the computation of the Scalar Kernel.
        Checks if K matrix and gradients have correct dimensions.
        """

        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='scalar')

        # Mock B_diag (diagonal of covariance)
        B_diag = tf.ones((self.nx,), dtype=tf.float32)

        K_val, grad_K = pff.compute_kernel_matrix(self.particles, B_diag)

        # For scalar kernel:
        # K_val should be [Np, Np, 1]
        # grad_K should be [Np, Np, Nx]
        self.assertEqual(K_val.shape, (self.np, self.np, 1), "Scalar Kernel K matrix shape incorrect.")
        self.assertEqual(grad_K.shape, (self.np, self.np, self.nx), "Scalar Kernel gradient shape incorrect.")

    def test_matrix_kernel_computation(self):
        """
        Test the computation of the Matrix-Valued Kernel.
        """

        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='matrix')

        B_diag = tf.ones((self.nx,), dtype=tf.float32)
        K_val, grad_K = pff.compute_kernel_matrix(self.particles, B_diag)

        # For matrix kernel:
        # K_val should be [Np, Np, Nx] (diagonal elements per dimension)
        self.assertEqual(K_val.shape, (self.np, self.np, self.nx), "Matrix Kernel K matrix shape incorrect.")

    def test_update_integration(self):
        """
        Integration Test: Run the full update loop.
        Ensures the pipeline runs end-to-end and particles are updated.
        """

        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='matrix')

        # Run update
        updated_particles = pff.update(self.particles, self.y_obs, n_steps=5, dt_s=0.01)

        # Check shape
        self.assertEqual(updated_particles.shape, self.particles.shape, "Updated particles lost shape integrity.")

        # Check that values actually changed (flow occurred)
        self.assertFalse(np.allclose(self.particles.numpy(), updated_particles.numpy()),
                         "Particles did not move after PFF update step.")

    def test_numerical_stability(self):
        """
        Test that the update does not produce NaNs or Infs.
        """

        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='scalar')
        updated_particles = pff.update(self.particles, self.y_obs, n_steps=5, dt_s=0.01)

        self.assertFalse(np.any(np.isnan(updated_particles.numpy())), "Update produced NaNs.")
        self.assertFalse(np.any(np.isinf(updated_particles.numpy())), "Update produced Infs.")


if __name__ == '__main__':
    unittest.main()