import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from src.models.linear_gaussian import LinearGaussianSSM
from src.filters.kalman import KalmanFilter
from src.utils import plot, metrics
tfd = tfp.distributions
tf.random.set_seed(42)
np.random.seed(42)

def main():
    # 1. Generate the true states and the corresponding observations.
    state_dim = 4  # n_x
    process_noise_dim = 4  # n_v
    obs_dim = 2  # n_y
    obs_noise_dim = 2  # n_w
    A = tf.eye(state_dim, dtype=tf.float32) + tf.random.normal((state_dim, state_dim), dtype=tf.float32) * 0.01
    B = tf.eye(process_noise_dim, dtype=tf.float32) * 0.3
    C = tf.random.normal((obs_dim, state_dim), dtype=tf.float32)
    D = tf.eye(obs_noise_dim, dtype=tf.float32) * 1e-3
    initial_mean = tf.zeros(state_dim, dtype=tf.float32)
    initial_cov = tf.eye(state_dim, dtype=tf.float32) * 1000
    num_timesteps = 100

    # Generate LGSSM data
    ssm = LinearGaussianSSM(state_dim, process_noise_dim, obs_dim, obs_noise_dim, A, B, C, D,
                            tf.eye(process_noise_dim, dtype=tf.float32), tf.eye(obs_noise_dim, dtype=tf.float32),
                            initial_mean, initial_cov)
    true_states, observations = ssm.generate_data(num_timesteps)

    # 2. Run the Kalman filter
    kf = KalmanFilter(state_dim, process_noise_dim, obs_dim, obs_noise_dim, num_timesteps, tf.constant(A), tf.constant(C), ssm.process_noise_cov, ssm.obs_noise_cov)

    # Standard Update
    print("Running Standard Kalman Filter.")
    kf.initialize(initial_mean, initial_cov)
    std_mean, std_cov = kf.run_filter(observations, 'standard')

    # Joseph Update
    print("Running Joseph Kalman Filter.")
    kf.initialize(initial_mean, initial_cov)
    joseph_mean, joseph_cov = kf.run_filter(observations, 'joseph')

    # use tfd.LinearGaussianStateSpaceModel as a benchmark
    print("Running TFP Benchmark...")
    proc_noise_scale = tf.linalg.cholesky(ssm.process_noise_cov)
    obs_noise_scale = tf.linalg.cholesky(ssm.obs_noise_cov)
    init_cov_scale = tf.linalg.cholesky(initial_cov)

    lgssm_tfp = tfd.LinearGaussianStateSpaceModel(
        num_timesteps=num_timesteps,
        transition_matrix=A,
        transition_noise=tfd.MultivariateNormalTriL(
            loc=tf.zeros(state_dim),
            scale_tril=proc_noise_scale
        ),
        observation_matrix=C,
        observation_noise=tfd.MultivariateNormalTriL(
            loc=tf.zeros(obs_dim),
            scale_tril=obs_noise_scale
        ),
        initial_state_prior=tfd.MultivariateNormalTriL(
            loc=initial_mean,
            scale_tril=init_cov_scale
        )
    )
    _, tf_mean, tf_cov, _, _, _, _ = lgssm_tfp.forward_filter(observations)


    # 3. Analyze filtering optimality and numerical stability
    # Compare standard update with TFP Benchmark
    diff_mean_std = np.linalg.norm(std_mean - tf_mean, axis=1)
    diff_cov_std = np.linalg.norm(std_cov - tf_cov, axis=(1, 2))
    print(f"Max Mean Difference (std-TFP): {np.max(diff_mean_std):.2e}")
    print(f"Max Covariance Difference (std-TFP): {np.max(diff_cov_std):.2e}")

    # Compare Joseph update with TFP Benchmark
    diff_mean_joseph = np.linalg.norm(joseph_mean - tf_mean, axis=1)
    diff_cov_joseph = np.linalg.norm(joseph_cov - tf_cov, axis=(1, 2))
    print(f"Max Mean Difference (Joseph-TFP): {np.max(diff_mean_joseph):.2e}")
    print(f"Max Covariance Difference (Joseph-TFP): {np.max(diff_cov_joseph):.2e}")

    # Symmetric analysis.
    std_sym_mean, std_sym_max = metrics.analyze_stability(std_cov)
    jos_sym_mean, jos_sym_max = metrics.analyze_stability(joseph_cov)
    tfp_sym_mean, tfp_sym_max = metrics.analyze_stability(tf_cov)
    print(f"Standard    - Avg Asymmetry: {std_sym_mean:.2e}, Max Asymmetry: {std_sym_max:.2e}")
    print(f"Joseph  - Avg Asymmetry: {jos_sym_mean:.2e}, Max Asymmetry: {jos_sym_max:.2e}")
    print(f"TFP - Avg Asymmetry: {tfp_sym_mean:.2e}, Max Asymmetry: {tfp_sym_max:.2e}")

    # Conditioning number.
    cond_std = metrics.condition_number(std_cov)
    cond_joseph = metrics.condition_number(joseph_cov)
    cond_tfp = metrics.condition_number(tf_cov)
    print(f"Max Condition Number (Standard): {np.max(cond_std):.2e}")
    print(f"Mean Condition Number (Standard): {np.mean(cond_std):.2e}")
    print(f"Max Condition Number (Joseph): {np.max(cond_joseph):.2e}")
    print(f"Mean Condition Number (Joseph): {np.mean(cond_joseph):.2e}")
    print(f"Max Condition Number (TFP): {np.max(cond_tfp):.2e}")
    print(f"Mean Condition Number (TFP): {np.mean(cond_tfp):.2e}")


    # Ratio of positive definite covariance matrix.
    pd_ratio_std = metrics.is_positive_definite(std_cov)
    print(f"Positive Definite Ratio (Standard): {pd_ratio_std:.2%}")
    pd_ratio_Joseph = metrics.is_positive_definite(joseph_cov)
    print(f"Positive Definite Ratio (Joseph): {pd_ratio_Joseph:.2%}")
    pd_ratio_tfp = metrics.is_positive_definite(tf_cov)
    print(f"Positive Definite Ratio (TFP): {pd_ratio_tfp:.2%}")

    # 4. Plot the comparison pictures
    # Compare the condition numbers of different methods
    plot.plot_condition_number(
        {'Standard': cond_std, 'Joseph': cond_joseph, 'TFP': cond_tfp},
        title="Condition Number Comparison",
        filename="kalman_condition_number.png"
    )

    # Compare the estimation error (L2 Norm)
    err_std = np.linalg.norm(std_mean - true_states[1:], axis=1)
    err_joseph = np.linalg.norm(joseph_mean - true_states[1:], axis=1)
    err_tfp = np.linalg.norm(tf_mean - true_states[1:], axis=1)

    plot.plot_error_metrics(
        {'Standard': err_std, 'Joseph': err_joseph, 'TFP': err_tfp},
        title="Estimation Error Norm (vs Truth)",
        filename="kalman_error.png"
    )

    plot.plot_error_metrics(
        {'Mean Diff': diff_mean_joseph, 'Cov Diff (Frobenius)': diff_cov_joseph},
        title="Deviation of Joseph updation from TFP Benchmark",
        filename="kalman_joseph_benchmark_diff.png"
    )

    plot.plot_error_metrics(
        {'Mean Deviation': diff_mean_std, 'Cov Deviation': diff_cov_std},
        title="Deviation of standard updation from TFP Benchmark",
        filename="kalman_std_benchmark_diff.png"
    )

if __name__ == "__main__":
    main()