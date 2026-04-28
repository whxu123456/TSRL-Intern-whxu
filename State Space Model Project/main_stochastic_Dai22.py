import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from src.models.stochastic_particle_EX1 import Dai22Example1Model
from src.filters.stochastic_particle_flow_Dai22 import ParticleFlowFilterDai22
from src.utils.plot import plot_homotopy_results


def run_static_example():
    """
    Reproduce Dai22 Example 1 as a single static Bayesian update.
    Also compute stiffness ratios for optimized and baseline homotopies.
    """
    model = Dai22Example1Model()
    pf = ParticleFlowFilterDai22(model, num_particles=50, num_steps=100)

    z = tf.constant([0.4754, 1.1868], dtype=tf.float32)

    prior_particles = model.sample_initial_particles(pf.N)
    prior_mean = model.prior_mean
    prior_cov = model.prior_cov

    # Optimized homotopy
    beta_opt, beta_dot_opt = pf.solve_optimal_beta(z, prior_mean, prior_cov)

    particles_opt = pf.run_flow(
        z=z,
        beta_vals=beta_opt,
        beta_dot_vals=beta_dot_opt,
        prior_particles=prior_particles,
        prior_mean=prior_mean,
        prior_cov=prior_cov
    )

    est_opt = tf.reduce_mean(particles_opt, axis=0)
    cov_opt = tfp.stats.covariance(particles_opt)

    # Baseline homotopy: beta(lambda) = lambda
    num_steps = pf.num_steps
    lambdas = np.linspace(0.0, 1.0, num_steps + 1).astype(np.float32)

    beta_base = lambdas
    beta_dot_base = np.ones_like(lambdas, dtype=np.float32)

    # Stiffness ratios
    R_stiff_opt = pf.compute_stiffness_ratio_path(
        z=z,
        beta_vals=beta_opt,
        prior_mean=prior_mean,
        prior_cov=prior_cov
    )

    R_stiff_base = pf.compute_stiffness_ratio_path(
        z=z,
        beta_vals=beta_base,
        prior_mean=prior_mean,
        prior_cov=prior_cov
    )

    print("\n--- Static Example Result ---")
    print("Estimate:", est_opt.numpy())
    print("Cov trace:", tf.linalg.trace(cov_opt).numpy())
    print("Max stiffness ratio (optimized):", np.max(R_stiff_opt))
    print("Max stiffness ratio (baseline):", np.max(R_stiff_base))

    return (
        particles_opt,
        est_opt,
        cov_opt,
        beta_opt,
        beta_dot_opt,
        R_stiff_opt,
        R_stiff_base
    )


def run_monte_carlo_table1(num_mc_runs=20, num_particles=100, num_steps=100):
    """
    Reproduce Dai22 Table 1: 20 Monte Carlo runs performance comparison.
    Compare baseline linear homotopy vs optimal homotopy.
    Args:
        num_mc_runs: Number of Monte Carlo runs, default 20 as in paper
        num_particles: Number of particles for filter
        num_steps: Number of flow steps
    Returns:
        results: Dictionary with all MC results and average
    """
    model = Dai22Example1Model()
    true_state = model.x_true

    rmse_baseline = []
    rmse_optimal = []
    trP_baseline = []
    trP_optimal = []

    print("\n" + "=" * 60)
    print("Running Monte Carlo experiments for Table 1...")
    print("=" * 60)
    print(
        f"{'MC':<3} | {'RMSE(Baseline)':<15} | {'RMSE(Optimal)':<15} | {'tr(P)(Baseline)':<15} | {'tr(P)(Optimal)':<15}")
    print("-" * 60)

    for mc_idx in range(num_mc_runs):
        # 1. Sample the observation
        z = model.sample_measurement()

        # 2. Initialize filter
        pf = ParticleFlowFilterDai22(model, num_particles=num_particles, num_steps=num_steps)
        prior_particles = model.sample_initial_particles(pf.N)
        prior_mean = model.prior_mean
        prior_cov = model.prior_cov

        # 3. Run with Optimal Homotopy
        beta_opt, beta_dot_opt = pf.solve_optimal_beta(z, prior_mean, prior_cov)
        particles_opt = pf.run_flow(
            z=z, beta_vals=beta_opt, beta_dot_vals=beta_dot_opt,
            prior_particles=prior_particles, prior_mean=prior_mean, prior_cov=prior_cov
        )
        est_opt = tf.reduce_mean(particles_opt, axis=0)
        cov_opt = tfp.stats.covariance(particles_opt)

        # 4. Run with Baseline Linear Homotopy
        beta_base, beta_dot_base = pf._linear_beta_schedule()
        particles_base = pf.run_flow(
            z=z, beta_vals=beta_base, beta_dot_vals=beta_dot_base,
            prior_particles=prior_particles, prior_mean=prior_mean, prior_cov=prior_cov
        )
        est_base = tf.reduce_mean(particles_base, axis=0)
        cov_base = tfp.stats.covariance(particles_base)

        # 5. Compute metrics
        rmse_o = tf.sqrt(tf.reduce_sum(tf.square(est_opt - true_state)))
        rmse_b = tf.sqrt(tf.reduce_sum(tf.square(est_base - true_state)))
        trP_o = tf.linalg.trace(cov_opt)
        trP_b = tf.linalg.trace(cov_base)

        # 6. Store results
        rmse_optimal.append(rmse_o.numpy())
        rmse_baseline.append(rmse_b.numpy())
        trP_optimal.append(trP_o.numpy())
        trP_baseline.append(trP_b.numpy())

        # 7. Print per-run result
        print(f"{mc_idx + 1:<3} | {rmse_b:<15.4f} | {rmse_o:<15.4f} | {trP_b:<15.2f} | {trP_o:<15.2f}")

    # Compute average results
    avg_rmse_b = np.mean(rmse_baseline)
    avg_rmse_o = np.mean(rmse_optimal)
    avg_trP_b = np.mean(trP_baseline)
    avg_trP_o = np.mean(trP_optimal)

    print("-" * 60)
    print(f"avg | {avg_rmse_b:<15.4f} | {avg_rmse_o:<15.4f} | {avg_trP_b:<15.2f} | {avg_trP_o:<15.2f}")
    print("=" * 60)
    print(f"Average RMSE (Baseline): {avg_rmse_b:.4f}")
    print(f"Average RMSE (Optimal): {avg_rmse_o:.4f}")
    print(f"Average tr(P) (Baseline): {avg_trP_b:.2f}")
    print(f"Average tr(P) (Optimal): {avg_trP_o:.2f}")
    print(f"RMSE Improvement: {(1 - avg_rmse_o / avg_rmse_b) * 100:.1f}%")
    print(f"tr(P) Improvement: {(1 - avg_trP_o / avg_trP_b) * 100:.1f}%")

    return {
        "rmse_baseline": rmse_baseline,
        "rmse_optimal": rmse_optimal,
        "trP_baseline": trP_baseline,
        "trP_optimal": trP_optimal,
        "average": {
            "rmse_b": avg_rmse_b,
            "rmse_o": avg_rmse_o,
            "trP_b": avg_trP_b,
            "trP_o": avg_trP_o
        }
    }


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    # 1. Static single-update example (Dai22 Example 1, for Fig.2)
    particles_opt, est_opt, cov_opt, beta_opt, beta_dot_opt, R_stiff_opt, R_stiff_base = run_static_example()

    lambdas = np.linspace(0, 1, len(beta_opt))
    plot_homotopy_results(
        lambdas=lambdas,
        beta_opt=np.array(beta_opt),
        beta_dot_opt=np.array(beta_dot_opt),
        R_stiff_opt=np.array(R_stiff_opt),
        R_stiff_base=np.array(R_stiff_base)
    )

    # 2. Monte Carlo experiment for Table 1
    run_monte_carlo_table1(num_mc_runs=20, num_particles=100, num_steps=100)


if __name__ == "__main__":
    main()