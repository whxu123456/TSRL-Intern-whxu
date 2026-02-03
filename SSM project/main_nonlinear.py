import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from src.models.stochastic_volatility import SVModel
from src.filters.EKF import EKF_SV, EKF_SV_Raw
from src.filters.UKF import UKF_SV, UKF_SV_Raw
from src.filters.particle_filter import ParticleFilter
from src.utils import plot, metrics

tf.random.set_seed(42)
np.random.seed(42)
def main():
    # 1. Initialize SV model and generate data
    T = 100
    model = SVModel(T=T)
    x_true, y_obs = model.generate_data()
    estimates = {}
    performance = {'RMSE': {}, 'Time': {}, 'Memory': {}}

    # Run EKF (log)
    print("Running EKF.")
    tracker = metrics.Tracker()
    tracker.start()
    ekf = EKF_SV(model)
    x_ekf, _ = ekf.run(y_obs)
    elapsed, mem = tracker.stop()
    estimates['EKF'] = x_ekf
    performance['RMSE']['EKF'] = metrics.calculate_rmse(x_true, x_ekf)
    performance['Time']['EKF'] = elapsed
    performance['Memory']['EKF'] = mem

    # Run EKF_raw (no log)
    tracker.start()
    ekf_raw = EKF_SV_Raw(model)
    x_ekf_raw, _ = ekf_raw.run(y_obs)
    elapsed, _ = tracker.stop()
    estimates['EKF (Raw)'] = x_ekf_raw
    performance['RMSE']['EKF (Raw)'] = metrics.calculate_rmse(x_true, x_ekf_raw)
    performance['Time']['EKF (Raw)'] = elapsed

    # run UKF (log)
    print("Running UKF.")
    tracker.start()
    ukf = UKF_SV(model)
    x_ukf, _ = ukf.run(y_obs)
    elapsed, mem = tracker.stop()
    estimates['UKF'] = x_ukf
    performance['RMSE']['UKF'] = metrics.calculate_rmse(x_true, x_ukf)
    performance['Time']['UKF'] = elapsed
    performance['Memory']['UKF'] = mem

    # run UKF_raw (no log)
    tracker.start()
    ukf_raw = UKF_SV_Raw(model)
    x_ukf_raw, _ = ukf_raw.run(y_obs)
    elapsed, _ = tracker.stop()
    estimates['UKF (Raw)'] = x_ukf_raw
    performance['RMSE']['UKF (Raw)'] = metrics.calculate_rmse(x_true, x_ukf_raw)
    performance['Time']['UKF (Raw)'] = elapsed


    # run Particle Filter
    print("Running Particle Filter.")
    tracker.start()
    pf = ParticleFilter(model, num_particles=1000)
    x_pf, ess_hist, _ = pf.run(y_obs)
    elapsed, mem = tracker.stop()
    estimates['PF'] = x_pf
    performance['RMSE']['PF'] = metrics.calculate_rmse(x_true, x_pf)
    performance['Time']['PF'] = elapsed
    performance['Memory']['PF'] = mem

    # 2. print the performance of the three algorithms
    print("\n" + "=" * 60)
    print(f"{'Algorithm':<10} | {'RMSE':<10} | {'Time (s)':<10} | {'Peak Mem (KB)':<15}")
    print("-" * 60)
    for alg in ['EKF', 'UKF', 'PF']:
        print(f"{alg:<10} | {performance['RMSE'][alg]:.4f}     | "
              f"{performance['Time'][alg]:.4f}     | {performance['Memory'][alg]:.2f}")
    print("=" * 60 + "\n")

    # 3. Plot comparison pictures.
    # 3.1 Trajectory comparison
    plot.plot_trajectory_comparison(
        x_true.numpy(),
        estimates,
        title='State Estimation (compared with true states)',
        filename='nonlinear_trajectory.png'
    )

    # 3.2 Compare estimation error
    errors = {name: np.abs(x_true.numpy() - est) for name, est in estimates.items()}
    plot.plot_error_metrics(
        errors,
        title='Absolute Estimation Error',
        filename='nonlinear_error.png'
    )

    # 3.3 Plot the effective sample size (ESS)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ess_hist, 'g-', label='ESS')
    ax.axhline(y=500, color='r', linestyle='--', label='Resample Threshold')
    ax.set_title('Particle Filter Degeneracy (ESS)')
    ax.set_ylabel('Effective Sample Size')
    ax.set_xlabel('Time Step')
    ax.legend()
    plot.save_or_show(fig, filename='nonlinear_ess.png')

    # 3.4 Trajectory Comparison (Focus on EKF vs Raw EKF)
    plot.plot_trajectory_comparison(
        x_true.numpy(),
        {k: v for k, v in estimates.items() if 'EKF' in k or 'PF' in k},
        title='Impact of Log-Transform: EKF vs Raw EKF',
        filename='ekf_raw_vs_log.png'
    )


if __name__ == "__main__":
    main()