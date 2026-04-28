import tensorflow as tf
import numpy as np
import pandas as pd
from src.models.acoustic import AcousticConfig, AcousticModel
from src.filters.EKF_acoustic import EKF_Predictor
from src.filters.acoustic_pfpf import AcousticPFPF
from src.utils import plot, metrics

tf.random.set_seed(11)
np.random.seed(11)

def run_single_experiment(mode, use_optimal_homotopy, cfg, model, x_true, z_meas):
    """
    Run a single PF-PF experiment with specified configuration.
    Args:
        mode: Flow mode ('EDH' or 'LEDH')
        use_optimal_homotopy: Enable Dai22 optimal homotopy or not
        cfg: AcousticConfig object
        model: AcousticModel object
        x_true: Ground truth state trajectory
        z_meas: Noisy measurement sequence
    Returns:
        results: Dictionary with experiment metrics
    """
    # Initialize trackers
    tracker = metrics.Tracker()
    tracker.start()

    ekf = EKF_Predictor(cfg, model)
    pf = AcousticPFPF(cfg, model, mode=mode, use_optimal_homotopy=use_optimal_homotopy)

    est_traj = []
    omat_errors = []

    # Run filter over time steps
    for t in range(cfg.T):
        # EKF prediction for predictive covariance
        P_pred = ekf.predict()
        # PF-PF update step
        z_curr = tf.constant(z_meas[t], dtype=tf.float32)
        x_est = pf.run_step(z_curr, P_pred)
        # EKF update for next step covariance
        ekf.update(z_curr)
        # Store results
        est_traj.append(x_est.numpy())
        omat_errors.append(metrics.calculate_omat(x_true[t], x_est.numpy(), cfg.n_targets))

    # Stop timing and memory tracking
    elapsed, peak_mem = tracker.stop()

    # Compile results
    experiment_name = f"{mode}_OptHomotopy" if use_optimal_homotopy else f"{mode}_Linear"
    results = {
        "name": experiment_name,
        "mode": mode,
        "use_optimal_homotopy": use_optimal_homotopy,
        "traj": np.array(est_traj),
        "omat": np.array(omat_errors),
        "avg_omat": np.mean(omat_errors),
        "final_omat": omat_errors[-1],
        "time_per_step": elapsed / cfg.T,
        "total_time": elapsed,
        "peak_mem_mb": peak_mem
    }

    print(f"Completed {experiment_name:20} | Avg OMAT: {results['avg_omat']:.4f} | Total Time: {elapsed:.2f}s")
    return results


def main():
    cfg = AcousticConfig()
    model = AcousticModel(cfg)
    print(f"Acoustic Tracking Scenario: T={cfg.T} steps, {cfg.n_targets} targets, {cfg.meas_dim} sensors")
    print(f"Particles: {cfg.n_particles}, Flow steps: {29}\n")

    # 2. Generate ground truth trajectory and noisy measurements
    x_true = np.zeros((cfg.T, cfg.state_dim), dtype=np.float32)
    z_meas = np.zeros((cfg.T, cfg.meas_dim), dtype=np.float32)

    x_curr = cfg.x0_truth
    for t in range(cfg.T):
        x_curr = model.propagate_truth(x_curr)
        x_true[t] = x_curr.numpy()
        z_noisy = model.sample_measurement(x_curr)
        z_meas[t] = z_noisy.numpy()

    # 3. Define experiment configurations (Li17 baseline vs Dai22 optimal)
    experiment_configs = [
        {"mode": "EDH", "use_optimal_homotopy": False},    # Li17 EDH baseline
        {"mode": "EDH", "use_optimal_homotopy": True},     # Dai22 Optimal EDH
        {"mode": "LEDH", "use_optimal_homotopy": False},   # Li17 LEDH baseline
        {"mode": "LEDH", "use_optimal_homotopy": True}     # Dai22 Optimal LEDH
    ]

    # 4. Run all experiments
    all_results = []
    for config in experiment_configs:
        result = run_single_experiment(
            mode=config["mode"],
            use_optimal_homotopy=config["use_optimal_homotopy"],
            cfg=cfg,
            model=model,
            x_true=x_true,
            z_meas=z_meas
        )
        all_results.append(result)

    # 5. Compile and print summary results
    print("Li17 Linear Homotopy vs Dai22 Optimal Homotopy")

    # Create summary dataframe
    summary_data = []
    for res in all_results:
        summary_data.append({
            "Configuration": res["name"],
            "Avg OMAT Error": f"{res['avg_omat']:.4f}",
            "Final OMAT Error": f"{res['final_omat']:.4f}",
            "Total Time (s)": f"{res['total_time']:.2f}",
            "Time per Step (ms)": f"{res['time_per_step']*1000:.1f}",
            "Peak Memory (MB)": f"{res['peak_mem_mb']:.1f}"
        })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))

    # Calculate performance improvement for LEDH
    ledh_linear = next(r for r in all_results if r["name"] == "LEDH_Linear")
    ledh_optimal = next(r for r in all_results if r["name"] == "LEDH_OptHomotopy")

    omat_improvement = (1 - ledh_optimal["avg_omat"] / ledh_linear["avg_omat"]) * 100
    time_overhead = (ledh_optimal["total_time"] / ledh_linear["total_time"] - 1) * 100

    print("Dai22 Optimal Homotopy vs Li17 LEDH Baseline")
    print(f"Li17 LEDH Baseline Avg OMAT: {ledh_linear['avg_omat']:.4f}")
    print(f"Dai22 Optimal LEDH Avg OMAT: {ledh_optimal['avg_omat']:.4f}")
    print(f"OMAT Tracking Error Reduction: {omat_improvement:.2f}%")
    print(f"Computational Overhead: {time_overhead:.2f}%")

    # 6. Generate comparison plots
    # OMAT error over time plot
    omat_dict = {res["name"]: res["omat"] for res in all_results}
    plot.plot_error_metrics(
        omat_dict,
        title="OMAT Tracking Error: Li17 Linear vs Dai22 Optimal Homotopy",
        filename="dai22_vs_li17_omat_comparison.png"
    )

    # Trajectory plot for optimal LEDH vs baseline LEDH
    traj_dict = {
        "Li17 LEDH Baseline": ledh_linear["traj"],
        "Dai22 Optimal LEDH": ledh_optimal["traj"]
    }
    plot.plot_acoustic_tracking(
        cfg.sensors_pos.numpy(),
        x_true,
        traj_dict,
        cfg.n_targets,
        filename="dai22_vs_li17_trajectory_comparison.png"
    )

    # 7. Comparison for EDH_OptHomotopy and LEDH_Linear
    edh_optimal = next(r for r in all_results if r["name"] == "EDH_OptHomotopy")

    print("Dai22 EDH Optimal Homotopy vs Li17 LEDH Baseline")
    print(f"Li17 LEDH Baseline Avg OMAT: {ledh_linear['avg_omat']:.4f}")
    print(f"Dai22 Optimal EDH Avg OMAT: {edh_optimal['avg_omat']:.4f}")

    omat_improvement_edh = (1 - edh_optimal["avg_omat"] / ledh_linear["avg_omat"]) * 100
    time_overhead_edh = (edh_optimal["total_time"] / ledh_linear["total_time"] - 1) * 100

    print(f"OMAT Tracking Error Reduction: {omat_improvement_edh:.2f}%")
    print(f"Computational Overhead: {time_overhead_edh:.2f}%")

    # Generate additional OMAT comparison plot (only EDH_Opt and LEDH_Linear)
    omat_dict_edh_vs_ledh = {
        "Li17 LEDH Baseline": ledh_linear["omat"],
        "Dai22 Optimal EDH": edh_optimal["omat"]
    }
    plot.plot_error_metrics(
        omat_dict_edh_vs_ledh,
        title="OMAT Tracking Error: Dai22 EDH Optimal vs Li17 LEDH Baseline",
        filename="dai22_edh_opt_vs_li17_ledh_omat_comparison.png"
    )

    # Generate additional trajectory comparison plot (only EDH_Opt and LEDH_Linear)
    traj_dict_edh_vs_ledh = {
        "Li17 LEDH Baseline": ledh_linear["traj"],
        "Dai22 Optimal EDH": edh_optimal["traj"]
    }
    plot.plot_acoustic_tracking(
        cfg.sensors_pos.numpy(),
        x_true,
        traj_dict_edh_vs_ledh,
        cfg.n_targets,
        filename="dai22_edh_opt_vs_li17_ledh_trajectory_comparison.png"
    )

    return all_results


if __name__ == "__main__":
    main()