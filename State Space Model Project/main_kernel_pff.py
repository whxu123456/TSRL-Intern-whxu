import tensorflow as tf
import numpy as np
from src.models.lorenz96 import Lorenz96
from src.filters.kernel_embedded import KernelEmbeddedPFF
from src.utils import plot

np.random.seed(42)
tf.random.set_seed(42)
def main():
    # 1. Experimental design
    Nx = 1000
    Np = 20
    Ny = Nx // 4  # 25% observation
    obs_indices = np.arange(3, Nx, 4)  # observe every 4 step

    model = Lorenz96(nx=Nx)

    # 2. Generate the true states
    print("Generating the true states.")
    x_truth = tf.ones((Nx,), dtype=tf.float32) * model.F
    x_truth = x_truth + tf.random.normal((Nx,), stddev=0.01)

    dt_model = 0.01
    for _ in range(1000):
        x_truth = model.rk4_step(x_truth, dt_model)

    # 3. create the prior and observations
    prior_std = 2
    obs_std = 0.5

    particles_prior = tf.stack([x_truth + tf.random.normal((Nx,), stddev=prior_std) for _ in range(Np)])

    # Generate the observation states
    y_truth = tf.gather(x_truth, obs_indices)
    y_obs = y_truth + tf.random.normal((Ny,), stddev=obs_std)

    # 4. Run the scalar and matrix-valued kernel PFF
    # Scalar Kernel
    print("Running PFF with Scalar Kernel...")
    pff_scalar = KernelEmbeddedPFF(Nx, Np, obs_indices, obs_std, kernel_type='scalar')
    particles_scalar = pff_scalar.update(particles_prior, y_obs, n_steps=100, dt_s=0.05)

    # Matrix-valued Kernel
    print("Running PFF with Matrix-Valued Kernel...")
    pff_matrix = KernelEmbeddedPFF(Nx, Np, obs_indices, obs_std, kernel_type='matrix')
    particles_matrix = pff_matrix.update(particles_prior, y_obs, n_steps=100, dt_s=0.05)

    # 5. Plot a picture similar to Figure 3 in Hu(21)

    # plot Matrix-valued Kernel
    plot.plot_lorenz_collapse(
        particles_prior.numpy(),
        particles_matrix.numpy(),
        x_truth.numpy(),
        obs_indices,
        title_suffix="(Matrix Kernel - Maintains Spread)",
        filename="lorenz_matrix_kernel.png"
    )

    # plot Scalar Kernel (collapse)
    plot.plot_lorenz_collapse(
        particles_prior.numpy(),
        particles_scalar.numpy(),
        x_truth.numpy(),
        obs_indices,
        title_suffix="(Scalar Kernel - Collapse)",
        filename="lorenz_scalar_kernel.png"
    )


if __name__ == "__main__":
    main()