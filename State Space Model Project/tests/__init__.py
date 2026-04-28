import unittest
import tensorflow as tf
import numpy as np
from src.filters.kernel_embedded import KernelEmbeddedPFF


class TestKernelEmbeddedPFF(unittest.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        self.nx = 20
        self.np = 10
        self.obs_indices = np.array([0, 5, 10, 15])
        self.obs_std = 0.1

    def test_matrix_kernel_update(self):
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='matrix')

        particles = tf.random.normal((self.np, self.nx))
        y_obs = tf.random.normal((len(self.obs_indices),))

        updated_particles = pff.update(particles, y_obs, n_steps=5)

        self.assertEqual(updated_particles.shape, (self.np, self.nx))
        self.assertFalse(np.allclose(particles.numpy(), updated_particles.numpy()), "Particles should move")

    def test_scalar_kernel_update(self):
        pff = KernelEmbeddedPFF(self.nx, self.np, self.obs_indices, self.obs_std, kernel_type='scalar')

        particles = tf.random.normal((self.np, self.nx))
        y_obs = tf.random.normal((len(self.obs_indices),))

        updated_particles = pff.update(particles, y_obs, n_steps=5)
        self.assertEqual(updated_particles.shape, (self.np, self.nx))


if __name__ == '__main__':
    unittest.main()