import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import time
from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel
from src.filters.EKF_nonlinear_ssm import EKF_NonlinearSSM
from src.filters.nonlinear_pfpf import NonlinearPFPF
from src.filters.differentiable_pfpf import DifferentiableNonlinearPFPF
tfd = tfp.distributions
tf.random.set_seed(42)
np.random.seed(42)

# 1. MCMC Core Utility Functions
def make_cfg_from_theta(theta, base_cfg):
    theta = np.asarray(theta, dtype=np.float32)

    cfg = NonlinearSSMConfig(T=base_cfg.T)
    cfg.alpha = np.float32(theta[0])
    cfg.beta = np.float32(theta[1])
    cfg.gamma = np.float32(theta[2])
    cfg.delta = np.float32(theta[3])
    cfg.sigma_v2 = np.float32(theta[4])
    cfg.sigma_w2 = np.float32(theta[5])

    cfg.sigma_v = np.sqrt(cfg.sigma_v2)
    cfg.sigma_w = np.sqrt(cfg.sigma_w2)

    cfg.Q_cov = tf.constant([[cfg.sigma_v2]], dtype=tf.float32)
    cfg.R_cov = tf.constant([[cfg.sigma_w2]], dtype=tf.float32)
    cfg.Q_chol = tf.linalg.cholesky(cfg.Q_cov)
    cfg.R_chol = tf.linalg.cholesky(cfg.R_cov)
    cfg.R_inv = tf.linalg.inv(cfg.R_cov)

    cfg.n_particles = base_cfg.n_particles

    return cfg


def theta_to_z(theta):
    theta = np.asarray(theta, dtype=np.float32)
    return np.array([theta[0], theta[1], theta[2], theta[3],
                     np.log(theta[4]), np.log(theta[5])], dtype=np.float32)


def z_to_theta(z):
    z = np.asarray(z, dtype=np.float32)
    return np.array([z[0], z[1], z[2], z[3],
                     np.exp(z[4]), np.exp(z[5])], dtype=np.float32)


@tf.function
def compute_log_prior_and_jac(z, cfg):
    alpha = z[0]
    beta = z[1]
    gamma = z[2]
    delta = z[3]
    sigma_v2 = tf.exp(z[4])
    sigma_w2 = tf.exp(z[5])

    theta = tf.stack([alpha, beta, gamma, delta, sigma_v2, sigma_w2])

    log_p = 0.0
    log_p += tfd.Normal(cfg.alpha, 1.0).log_prob(alpha)
    log_p += tfd.Normal(cfg.beta, 5.0).log_prob(beta)
    log_p += tfd.Normal(cfg.gamma, 2.0).log_prob(gamma)
    log_p += tfd.Normal(cfg.delta, 0.5).log_prob(delta)
    log_p += tfd.InverseGamma(2.0, 10.0).log_prob(sigma_v2)
    log_p += tfd.InverseGamma(2.0, 1.0).log_prob(sigma_w2)

    log_jac = z[4] + z[5]

    return theta, log_p, log_jac


def log_prior_z(z, base_cfg):
    theta = z_to_theta(z)

    if theta[4] <= 0.0 or theta[5] <= 0.0:
        return -np.inf

    cfg_local = make_cfg_from_theta(theta, base_cfg)
    theta_tf = tf.convert_to_tensor(theta, dtype=tf.float32)

    lp = log_prior(theta_tf, cfg_local).numpy()
    lp += z[4] + z[5]

    if not np.isfinite(lp):
        return -np.inf

    return float(lp)


def pmmh_loglik(theta, base_cfg, y_obs):
    if not np.all(np.isfinite(theta)):
        return -np.inf
    if theta[4] <= 0.0 or theta[5] <= 0.0:
        return -np.inf

    cfg_local = make_cfg_from_theta(theta, base_cfg)
    model_local = NonlinearSSMModel(cfg_local)
    ekf_local = EKF_NonlinearSSM(cfg_local, model_local)
    pf = NonlinearPFPF(cfg_local, model_local, ekf_local, flow_method="LEDH")

    _, _, loglik, _ = pf.run_filter_with_loglik(y_obs)

    if not np.isfinite(loglik):
        return -np.inf

    return float(loglik)


def log_prior(theta, cfg):
    """
    Weakly informative prior distribution for model parameters
    Args:
        theta: [alpha, beta, gamma, delta, sigma_v2, sigma_w2]
        cfg: NonlinearSSMConfig instance
    Returns:
        log_prior: scalar log prior probability
    """
    alpha, beta, gamma, delta, sigma_v2, sigma_w2 = theta

    log_p = 0.0
    log_p += tfd.Normal(cfg.alpha, 1.0).log_prob(alpha)
    log_p += tfd.Normal(cfg.beta, 5.0).log_prob(beta)
    log_p += tfd.Normal(cfg.gamma, 2.0).log_prob(gamma)
    log_p += tfd.Normal(cfg.delta, 0.5).log_prob(delta)
    log_p += tfd.InverseGamma(2.0, 10.0).log_prob(sigma_v2)
    log_p += tfd.InverseGamma(2.0, 1.0).log_prob(sigma_w2)

    return log_p


def run_pmmh(y_obs, cfg, model=None, n_samples=500, burnin=100):
    """
    PMMH in transformed parameter space:
    z = [alpha, beta, gamma, delta, log_sigma_v2, log_sigma_w2]
    """
    y_obs = np.asarray(y_obs, dtype=np.float32)

    theta_init = np.array([0.6, 20.0, 7.0, 1.0, 12.0, 1.2], dtype=np.float32)
    z_current = theta_to_z(theta_init)
    theta_current = z_to_theta(z_current)

    samples = np.zeros((n_samples, 6), dtype=np.float32)

    log_lik_current = pmmh_loglik(theta_current, cfg, y_obs)
    log_post_current = log_lik_current + log_prior_z(z_current, cfg)

    if not np.isfinite(log_post_current):
        raise RuntimeError("Initial PMMH log posterior is not finite.")

    proposal_scales = np.array([0.02, 0.5, 0.15, 0.03, 0.08, 0.08], dtype=np.float32)
    proposal_cov = np.diag(proposal_scales ** 2)

    n_accepted = 0
    start_time = time.time()

    for i in range(n_samples):
        z_proposed = z_current + np.random.multivariate_normal(
            np.zeros(6), proposal_cov
        ).astype(np.float32)

        theta_proposed = z_to_theta(z_proposed)

        if theta_proposed[4] > 1e4 or theta_proposed[5] > 1e4:
            log_post_proposed = -np.inf
        else:
            log_prior_proposed = log_prior_z(z_proposed, cfg)
            if np.isfinite(log_prior_proposed):
                log_lik_proposed = pmmh_loglik(theta_proposed, cfg, y_obs)
                log_post_proposed = log_lik_proposed + log_prior_proposed
            else:
                log_post_proposed = -np.inf

        log_alpha = log_post_proposed - log_post_current

        if np.log(np.random.rand()) < log_alpha:
            z_current = z_proposed
            theta_current = theta_proposed
            log_post_current = log_post_proposed
            n_accepted += 1

        samples[i] = theta_current

        if (i + 1) % 20 == 0:
            print(
                f"PMMH {i + 1}/{n_samples}, "
                f"acc={n_accepted / (i + 1):.3f}, "
                f"logpost={log_post_current:.2f}, "
                f"theta={theta_current}"
            )

    runtime = time.time() - start_time
    return samples[burnin:], n_accepted / n_samples, runtime


def run_hmc(y_obs, cfg, model, n_samples=40, burnin=10):
    """
    HMC with conservative parameters
    """
    y_obs_tensor = tf.convert_to_tensor(y_obs, dtype=np.float32)

    model_local = NonlinearSSMModel(cfg)
    ekf_local = EKF_NonlinearSSM(cfg, model_local)
    dpf = DifferentiableNonlinearPFPF(
        cfg,
        model_local,
        ekf_local,
        flow_method="LEDH",
        epsilon=0.5,
    )

    def target_log_prob(z):
        alpha = z[0]
        beta = z[1]
        gamma = z[2]
        delta = z[3]
        sigma_v2 = tf.exp(z[4])
        sigma_w2 = tf.exp(z[5])

        theta = tf.stack([alpha, beta, gamma, delta, sigma_v2, sigma_w2])

        model_local.cfg.alpha = alpha
        model_local.cfg.beta = beta
        model_local.cfg.gamma = gamma
        model_local.cfg.delta = delta
        model_local.cfg.sigma_v2 = sigma_v2
        model_local.cfg.sigma_w2 = sigma_w2
        model_local.cfg.Q_cov = tf.reshape(sigma_v2, [1, 1])
        model_local.cfg.R_cov = tf.reshape(sigma_w2, [1, 1])

        log_lik = dpf.compute_differentiable_log_likelihood(y_obs_tensor, theta)

        log_p = 0.0
        log_p += tfd.Normal(cfg.alpha, 1.0).log_prob(alpha)
        log_p += tfd.Normal(cfg.beta, 5.0).log_prob(beta)
        log_p += tfd.Normal(cfg.gamma, 2.0).log_prob(gamma)
        log_p += tfd.Normal(cfg.delta, 0.5).log_prob(delta)
        log_p += tfd.InverseGamma(2.0, 10.0).log_prob(sigma_v2)
        log_p += tfd.InverseGamma(2.0, 1.0).log_prob(sigma_w2)

        log_jac = z[4] + z[5]

        return log_lik + log_p + log_jac

    initial_theta = tf.constant([0.6, 20.0, 7.0, 1.0, 12.0, 1.2], dtype=tf.float32)
    initial_state = tf.stack([
        initial_theta[0],
        initial_theta[1],
        initial_theta[2],
        initial_theta[3],
        tf.math.log(initial_theta[4]),
        tf.math.log(initial_theta[5]),
    ])

    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=0.005,
        num_leapfrog_steps=2
    )

    start_time = time.time()

    print("Starting HMC sampling...")
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=n_samples,
        num_burnin_steps=burnin,
        current_state=initial_state,
        kernel=hmc_kernel,
        trace_fn=lambda _, pkr: pkr.is_accepted,
        seed=42
    )
    runtime = time.time() - start_time

    acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32)).numpy()
    z_samples = samples.numpy()

    theta_samples = np.zeros_like(z_samples, dtype=np.float32)
    theta_samples[:, 0] = z_samples[:, 0]
    theta_samples[:, 1] = z_samples[:, 1]
    theta_samples[:, 2] = z_samples[:, 2]
    theta_samples[:, 3] = z_samples[:, 3]
    theta_samples[:, 4] = np.exp(z_samples[:, 4])
    theta_samples[:, 5] = np.exp(z_samples[:, 5])

    return theta_samples, acceptance_rate, runtime


def compute_metrics(samples, true_params, runtime):
    """
    Manually computes ESS for robustness
    """
    samples = np.asarray(samples, dtype=np.float32)

    def compute_ess(x):
        n = len(x)
        if n < 2:
            return np.nan

        mean_x = np.mean(x)
        var_x = np.var(x, ddof=1)

        if var_x < 1e-10:
            return n

        autocorr = np.zeros(n)
        for k in range(n):
            if k == 0:
                autocorr[k] = 1.0
            else:
                cov = np.mean((x[:-k] - mean_x) * (x[k:] - mean_x))
                autocorr[k] = cov / var_x

        positive_autocorr = autocorr[autocorr > 0]
        if len(positive_autocorr) < 2:
            trunc = 1
        else:
            first_negative = np.where(autocorr < 0)[0]
            if len(first_negative) > 0:
                trunc = first_negative[0]
            else:
                trunc = min(20, n // 2)

        ess = n / (1 + 2 * np.sum(autocorr[1:trunc]))
        return ess

    n_params = samples.shape[1]
    ess_per_param = np.zeros(n_params)
    for i in range(n_params):
        ess_per_param[i] = compute_ess(samples[:, i])

    mean_ess = np.nanmean(ess_per_param)
    ess_per_second = mean_ess / runtime

    param_means = np.mean(samples, axis=0)
    bias = param_means - true_params
    rmse = np.sqrt(np.mean((samples - true_params) ** 2, axis=0))

    return {
        "mean_ess": mean_ess,
        "ess_per_second": ess_per_second,
        "mean_rhat": np.nan,
        "bias": bias,
        "rmse": rmse,
        "runtime": runtime,
    }


# 2. Quantitative Diagnostics
def run_diagnostics(y_obs, cfg, true_params):
    """
    Run comprehensive diagnostics to quantitatively analyze:
    1. Differentiability-Bias Trade-off
    2. OT Regularization Effects (epsilon sweep)
    3. Gradient Stability & Variance
    4. Computational Complexity (scaling test)
    """
    print("\n[Diagnostic 1/4] Differentiability-Bias Trade-off")
    model = NonlinearSSMModel(cfg)
    ekf = EKF_NonlinearSSM(cfg, model)

    pf_standard = NonlinearPFPF(cfg, model, ekf, flow_method="LEDH")
    _, _, loglik_standard, _ = pf_standard.run_filter_with_loglik(y_obs)

    epsilons = [0.1, 0.5, 1.0, 2.0]
    dpf_logliks = []
    dpf_biases = []

    for eps in epsilons:
        dpf = DifferentiableNonlinearPFPF(cfg, model, ekf, flow_method="LEDH", epsilon=eps)
        loglik_dpf = dpf.compute_differentiable_log_likelihood(y_obs, tf.convert_to_tensor(true_params))
        dpf_logliks.append(loglik_dpf.numpy())
        bias = abs(loglik_dpf.numpy() - loglik_standard)
        dpf_biases.append(bias)
        print(f"  epsilon={eps:.1f} | Log-likelihood: {loglik_dpf.numpy():.2f} | Bias vs Standard PF: {bias:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, dpf_biases, 'o-', linewidth=2, markersize=8, color='red')
    plt.xlabel("OT Regularization Parameter (epsilon)", fontsize=12)
    plt.ylabel("Absolute Bias in Log-likelihood", fontsize=12)
    plt.title("Differentiability-Bias Trade-off: Bias vs Epsilon", fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('diagnostic_bias_vs_epsilon.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n[Diagnostic 2/4] OT Regularization Effects: Particle Smoothness")
    dpf = DifferentiableNonlinearPFPF(cfg, model, ekf, flow_method="LEDH", epsilon=0.5)
    dpf.compute_differentiable_log_likelihood(y_obs, tf.convert_to_tensor(true_params))

    print(f"  Final particle set shape: {dpf.particles.shape}")
    print(f"  Final particle weights: {dpf.weights.numpy()[:5]}...")

    plt.figure(figsize=(10, 6))
    plt.scatter(range(cfg.n_particles), dpf.particles.numpy().flatten(),
                s=100, alpha=0.6, c=dpf.weights.numpy(), cmap='viridis')
    plt.colorbar(label="Particle Weight")
    plt.xlabel("Particle Index", fontsize=12)
    plt.ylabel("Particle Value (Latent State)", fontsize=12)
    plt.title("OT Regularization Effect: Final Particles", fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig('diagnostic_particles.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n[Diagnostic 3/4] Gradient Stability & Variance")
    N_GRADIENT_RUNS = 5
    gradients = []

    for i in range(N_GRADIENT_RUNS):
        print(f"  Gradient run {i + 1}/{N_GRADIENT_RUNS}...")

        model_local = NonlinearSSMModel(cfg)
        ekf_local = EKF_NonlinearSSM(cfg, model_local)
        dpf_local = DifferentiableNonlinearPFPF(cfg, model_local, ekf_local, flow_method="LEDH", epsilon=0.5)

        z_true = theta_to_z(true_params)
        z_tf = tf.convert_to_tensor(z_true, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(z_tf)
            theta, log_p, log_jac = compute_log_prior_and_jac(z_tf, cfg)

            model_local.cfg.alpha = theta[0]
            model_local.cfg.beta = theta[1]
            model_local.cfg.gamma = theta[2]
            model_local.cfg.delta = theta[3]
            model_local.cfg.sigma_v2 = theta[4]
            model_local.cfg.sigma_w2 = theta[5]
            model_local.cfg.Q_cov = tf.reshape(theta[4], [1, 1])
            model_local.cfg.R_cov = tf.reshape(theta[5], [1, 1])

            log_lik = dpf_local.compute_differentiable_log_likelihood(y_obs, theta)
            target = log_lik + log_p + log_jac

        grad = tape.gradient(target, z_tf)
        gradients.append(grad.numpy())

    gradients = np.array(gradients)
    grad_mean = np.mean(gradients, axis=0)
    grad_std = np.std(gradients, axis=0)
    grad_variance = np.var(gradients, axis=0)

    print("\n  Gradient Statistics:")
    param_names = ["alpha", "beta", "gamma", "delta", "log_sigma_v2", "log_sigma_w2"]
    for i in range(6):
        print(
            f"    {param_names[i]:<15} | Mean={grad_mean[i]:.4f} | Std={grad_std[i]:.4f} | Var={grad_variance[i]:.6f}")

    plt.figure(figsize=(12, 6))
    x = np.arange(6)
    width = 0.35
    plt.bar(x - width / 2, grad_mean, width, label='Gradient Mean', alpha=0.7)
    plt.bar(x + width / 2, grad_std, width, label='Gradient Std (Variability)', alpha=0.7, color='red')
    plt.xticks(x, param_names, fontsize=11)
    plt.ylabel("Gradient Value", fontsize=12)
    plt.title("Gradient Stability: Mean & Standard Deviation Across Runs", fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('diagnostic_gradient_variance.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n[Diagnostic 4/4] Computational Complexity: Scaling with Particle Count")
    particle_counts = [5, 10, 15, 20]
    runtimes_pf = []
    runtimes_dpf = []

    for N in particle_counts:
        print(f"  Testing N={N} particles...")
        cfg_test = NonlinearSSMConfig(T=cfg.T)
        cfg_test.n_particles = N
        model_test = NonlinearSSMModel(cfg_test)
        ekf_test = EKF_NonlinearSSM(cfg_test, model_test)

        pf_test = NonlinearPFPF(cfg_test, model_test, ekf_test, flow_method="LEDH")
        start = time.time()
        pf_test.run_filter_with_loglik(y_obs)
        runtimes_pf.append(time.time() - start)

        dpf_test = DifferentiableNonlinearPFPF(cfg_test, model_test, ekf_test, flow_method="LEDH", epsilon=0.5)
        theta_test = tf.convert_to_tensor(true_params, dtype=tf.float32)
        start = time.time()
        dpf_test.compute_differentiable_log_likelihood(y_obs, theta_test)
        runtimes_dpf.append(time.time() - start)

    print("\n  Runtime Results:")
    for i, N in enumerate(particle_counts):
        print(
            f"    N={N:2d} | PF: {runtimes_pf[i]:.2f}s | DPF: {runtimes_dpf[i]:.2f}s | Ratio: {runtimes_dpf[i] / runtimes_pf[i]:.1f}x")

    plt.figure(figsize=(10, 6))
    plt.plot(particle_counts, runtimes_pf, 'o-', linewidth=2, markersize=8, label='Standard PF (O(N))', color='blue')
    plt.plot(particle_counts, runtimes_dpf, 's-', linewidth=2, markersize=8, label='DPF (O(N²))', color='red')

    x_theory = np.linspace(particle_counts[0], particle_counts[-1], 100)
    plt.plot(x_theory, runtimes_pf[0] * (x_theory / particle_counts[0]), '--', color='blue', alpha=0.5,
             label='Theoretical O(N)')
    plt.plot(x_theory, runtimes_dpf[0] * (x_theory / particle_counts[0]) ** 2, '--', color='red', alpha=0.5,
             label='Theoretical O(N²)')

    plt.xlabel("Number of Particles (N)", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Computational Complexity: Scaling with Particle Count", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('diagnostic_complexity_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Li(17) PF-PF + Corenflos(21) Differentiable Resampling: HMC vs PMMH Comparison")

    cfg = NonlinearSSMConfig(T=5)
    cfg.n_particles = 5
    model = NonlinearSSMModel(cfg)

    print(f"\n[1/4] Generating synthetic data (T={cfg.T} steps)...")
    x_true, y_obs = model.generate_true_trajectory()
    true_params = np.array([
        cfg.alpha, cfg.beta, cfg.gamma, cfg.delta,
        cfg.sigma_v2, cfg.sigma_w2
    ], dtype=np.float32)
    print(f"True parameters: alpha={cfg.alpha:.2f}, beta={cfg.beta:.2f}, gamma={cfg.gamma:.2f}, delta={cfg.delta:.2f}")

    print("\n[2/4] Running PMMH...")
    pmmh_samples, pmmh_acc, pmmh_runtime = run_pmmh(y_obs, cfg, model, n_samples=20, burnin=5)

    print("\n[3/4] Running HMC with Differentiable PF-PF...")
    hmc_samples, hmc_acc, hmc_runtime = run_hmc(y_obs, cfg, model, n_samples=20, burnin=5)

    print("\n[4/4] Computing performance metrics...")
    pmmh_metrics = compute_metrics(pmmh_samples, true_params, pmmh_runtime)
    pmmh_metrics["acceptance_rate"] = pmmh_acc

    hmc_metrics = compute_metrics(hmc_samples, true_params, hmc_runtime)
    hmc_metrics["acceptance_rate"] = hmc_acc

    print("Performance Comparison: PMMH vs HMC")
    print(f"{'Metric':<25} {'PMMH':<15} {'HMC':<15}")
    print(f"{'Acceptance Rate':<25} {pmmh_metrics['acceptance_rate']:<15.3f} {hmc_metrics['acceptance_rate']:<15.3f}")
    print(f"{'Mean Chain ESS':<25} {pmmh_metrics['mean_ess']:<15.1f} {hmc_metrics['mean_ess']:<15.1f}")
    print(f"{'ESS per Second':<25} {pmmh_metrics['ess_per_second']:<15.2f} {hmc_metrics['ess_per_second']:<15.2f}")
    print(f"{'Total Runtime (s)':<25} {pmmh_metrics['runtime']:<15.2f} {hmc_metrics['runtime']:<15.2f}")
    print(f"{'Mean Parameter RMSE':<25} {np.mean(pmmh_metrics['rmse']):<15.3f} {np.mean(hmc_metrics['rmse']):<15.3f}")

    run_diagnostics(y_obs, cfg, true_params)