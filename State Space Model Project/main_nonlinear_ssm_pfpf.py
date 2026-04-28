import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel
from src.filters.EKF_nonlinear_ssm import EKF_NonlinearSSM
from src.filters.nonlinear_pfpf import NonlinearPFPF


def full_run_test():
    """
    End-to-end complete test:
    1. Generate true trajectory and observations
    2. Run EKF, PF-PF(EDH), PF-PF(LEDH)
    3. Output performance metric comparison
    4. Plot state estimation comparison
    """
    # 1. Model initialization
    np.random.seed(42)
    tf.random.set_seed(42)
    cfg = NonlinearSSMConfig(T=100)
    cfg.n_particles = 500  # Default particle count from Li (2017) paper
    model = NonlinearSSMModel(cfg)

    # 2. Generate true trajectory and observations
    x_true, y_obs = model.generate_true_trajectory()
    print(f"Generated true trajectory length: {cfg.T} steps")
    print(f"State dimension: {cfg.state_dim}, Measurement dimension: {cfg.meas_dim}")

    # 3. Initialize filters
    ekf = EKF_NonlinearSSM(cfg, model)
    pfpf_edh = NonlinearPFPF(cfg, model, EKF_NonlinearSSM(cfg, model), flow_method='EDH')
    pfpf_ledh = NonlinearPFPF(cfg, model, EKF_NonlinearSSM(cfg, model), flow_method='LEDH')

    # 4. Run filtering
    print("Running EKF...")
    x_ekf, time_ekf = ekf.run(y_obs)
    rmse_ekf = np.sqrt(np.mean((x_ekf - x_true) ** 2))

    print("Running PF-PF (EDH)...")
    x_edh, ess_edh, time_edh = pfpf_edh.run_filter(y_obs)
    rmse_edh = np.sqrt(np.mean((x_edh - x_true) ** 2))
    mean_ess_edh = np.mean(ess_edh)

    print("Running PF-PF (LEDH)...")
    x_ledh, ess_ledh, time_ledh = pfpf_ledh.run_filter(y_obs)
    rmse_ledh = np.sqrt(np.mean((x_ledh - x_true) ** 2))
    mean_ess_ledh = np.mean(ess_ledh)

    # 5. Output performance metrics
    print("Performance Metric Comparison:")
    print(f"{'Algorithm':<15} {'RMSE':<10} {'Mean ESS':<10} {'Runtime (s)':<10}")
    print(f"{'EKF':<15} {rmse_ekf:<10.4f} {'N/A':<10} {time_ekf:<10.4f}")
    print(f"{'PF-PF (EDH)':<15} {rmse_edh:<10.4f} {mean_ess_edh:<10.1f} {time_edh:<10.4f}")
    print(f"{'PF-PF (LEDH)':<15} {rmse_ledh:<10.4f} {mean_ess_ledh:<10.1f} {time_ledh:<10.4f}")

    # 6. Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(x_true, 'k-', label='True State', linewidth=2, alpha=0.7)
    plt.plot(x_ekf, 'b--', label='EKF Estimate', alpha=0.8)
    plt.plot(x_edh, 'g-.', label='PF-PF (EDH) Estimate', alpha=0.8)
    plt.plot(x_ledh, 'r:', label='PF-PF (LEDH) Estimate', alpha=0.8)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('State Value x_t', fontsize=12)
    plt.title('Nonlinear State Space Model State Estimation Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('pfpf_nonlinear_state_est.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot ESS sequence
    plt.figure(figsize=(12, 4))
    plt.plot(ess_edh, 'g-', label='PF-PF (EDH) ESS', alpha=0.8)
    plt.plot(ess_ledh, 'r-', label='PF-PF (LEDH) ESS', alpha=0.8)
    plt.axhline(y=cfg.n_particles/2, color='k', linestyle='--', label='Resampling Threshold')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Effective Sample Size (ESS)', fontsize=12)
    plt.title('PF-PF Filter Effective Sample Size Evolution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('pfpf_nonlinear_ess.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'rmse_ekf': rmse_ekf,
        'rmse_edh': rmse_edh,
        'rmse_ledh': rmse_ledh,
        'time_ekf': time_ekf,
        'time_edh': time_edh,
        'time_ledh': time_ledh,
        'mean_ess_edh': mean_ess_edh,
        'mean_ess_ledh': mean_ess_ledh
    }


if __name__ == "__main__":
    results = full_run_test()