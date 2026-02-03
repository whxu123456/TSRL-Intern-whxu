import tensorflow as tf
import numpy as np
from src.models.acoustic import AcousticConfig, AcousticModel
from src.filters.EKF import EKF_Predictor
from src.filters.particle_flow_filter import AcousticPFPF
from src.utils import plot, metrics

tf.random.set_seed(42)
np.random.seed(42)
def main():
    # 1. Generate the data of Acoustic tracking model (trajectory and measurements)
    cfg = AcousticConfig()
    model = AcousticModel(cfg)
    x_true = np.zeros((cfg.T, cfg.state_dim), dtype=np.float32)
    z_meas = np.zeros((cfg.T, cfg.meas_dim), dtype=np.float32)

    x_curr = cfg.x0_mean
    for t in range(cfg.T):
        x_curr = model.transition(tf.expand_dims(x_curr, 0), noise=True)
        x_curr = tf.squeeze(x_curr, 0)
        # limit the trajectory inside the sensors
        x_curr = model.limit_boundary(x_curr)
        x_true[t] = x_curr.numpy()
        z_clean = model.measurement(tf.expand_dims(x_curr, 0))
        z_noisy = z_clean + tf.random.normal(tf.shape(z_clean)) * cfg.sigma_w
        z_meas[t] = tf.squeeze(z_noisy, 0).numpy()

    # 3. Run the EKF filter
    modes = ['LEDH', 'EDH']
    results = {}

    for mode in modes:
        print(f"Running PF-PF ({mode}).")
        tracker = metrics.Tracker()
        tracker.start()

        ekf = EKF_Predictor(cfg, model)
        pf = AcousticPFPF(cfg, model, mode=mode)

        est_traj = []
        omat_errors = []

        for t in range(cfg.T):
            # 1. EKF prediction for P
            P_pred = ekf.predict()
            # 2. PF filter
            z_curr = tf.constant(z_meas[t], dtype=tf.float32)
            x_est = pf.run_step(z_curr, P_pred)
            # 3. EKF update
            ekf.update(z_curr)
            est_traj.append(x_est.numpy())
            # 4. calcuate omat
            omat_errors.append(metrics.calculate_omat(x_true[t], x_est.numpy(), cfg.n_targets))

        elapsed, peak_mem = tracker.stop()

        results[mode] = {
            'traj': np.array(est_traj),
            'omat': np.array(omat_errors),
            'time': elapsed,
            'mem': peak_mem
        }
        print(f"Mode: {mode} | Avg OMAT: {np.mean(omat_errors):.4f} | Time: {elapsed:.2f}s")

    # 4. Plot the pictures
    # OMAT pic
    omat_dict = {mode: res['omat'] for mode, res in results.items()}
    plot.plot_error_metrics(omat_dict, title="OMAT Error over Time", filename="acoustic_omat.png")

    # Tajectory picture of LEDH Algorithm
    plot.plot_acoustic_tracking(
        cfg.sensors_pos.numpy(),
        x_true,
        {'LEDH': results['LEDH']['traj']},
        cfg.n_targets,
        filename="acoustic_trajectory.png"
    )


if __name__ == "__main__":
    main()
