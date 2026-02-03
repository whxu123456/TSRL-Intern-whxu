import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.models.stochastic_volatility import SVModel
from src.filters.particle_flow_filter import SVParticleFlowFilter
from src.utils import plot, metrics

np.random.seed(11)
tf.random.set_seed(11)
def main():
    # 1. Generate the data from stochastic volatility model
    T = 100
    model = SVModel(T=T)
    x_true, y_obs = model.generate_data()

    # 2. Run Filters
    methods = ['EDH_Log', 'EDH_Exact', 'LEDH_Exact', 'Kernel']
    results = {}  # For the estimated mean of states
    # Store full history of diagnostics for plotting
    diag_flow_mag = {m: [] for m in methods}
    diag_cond_num = {m: [] for m in methods}

    for m in methods:
        print(f"Running {m}.")
        pf = SVParticleFlowFilter(model, num_particles=100, num_steps=50, method=m)
        est_means = []

        for t in range(T):
            pf.predict()
            pf.update(y_obs[t])
            est_means.append(tf.reduce_mean(pf.particles).numpy())
            diag_flow_mag[m].extend(pf.flow_mags)
            diag_cond_num[m].extend(pf.condition_numbers)
        results[m] = np.array(est_means)

    # 3. Calculate RMSE
    rmse_dict = {}
    for m in methods:
        rmse_dict[m] = metrics.calculate_rmse(x_true, results[m])
        print(f"Method: {m:<10} | RMSE: {rmse_dict[m]:.4f}")

    # 4. Plots
    # 4.1 trajectory
    plot.plot_trajectory_comparison(
        x_true.numpy(),
        results,
        title='State Estimation (Stochastic Volatility)',
        filename='dkff_trajectory.png'
    )

    # 4.2 Stability Diagnostics (Flow Magnitude ONLY)
    # Since SV is 1D, Condition Number is trivially 1.0. We only analyze Flow Magnitude.
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['b', 'g', 'r', 'orange']  # 预定义颜色以便统一
    style_map = {m: c for m, c in zip(methods, colors)}

    for m in methods:
        data = np.array(diag_flow_mag[m])
        # 为了防止 log(0) 报错，加一个极小值 eps
        data = np.maximum(data, 1e-10)

        ax.plot(data, label=m, color=style_map.get(m, 'k'), alpha=0.7, linewidth=1)

    ax.set_title('Combined Flow Magnitude ||f|| (Stability Comparison)')
    ax.set_ylabel('Magnitude (log scale)')
    ax.set_xlabel('Cumulative Flow Steps (Time * Steps)')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    plot.save_or_show(fig, filename='dkff_stability_flow_mag_combined.png')

    # 4.3 Individual Stability Diagnostics (Subplots)
    n_methods = len(methods)
    rows = 2
    cols = 2
    if n_methods > 4: rows = (n_methods + 1) // 2

    fig_sub, axes_sub = plt.subplots(rows, cols, figsize=(14, 10), sharex=True)
    axes_flat = axes_sub.flatten()

    for i, m in enumerate(methods):
        if i >= len(axes_flat): break

        ax = axes_flat[i]
        data = np.array(diag_flow_mag[m])
        data = np.maximum(data, 1e-10)  # Safety for log scale

        # 使用与总图相同的颜色
        c = style_map.get(m, 'k')

        ax.plot(data, label=m, color=c, alpha=0.8, linewidth=1)

        ax.set_title(f'{m} Flow Magnitude')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend(loc='upper right')

        # 只在最左侧的图显示 Y 轴标签，最底部的图显示 X 轴标签
        if i % cols == 0:
            ax.set_ylabel('Magnitude (log)')
        if i >= (rows - 1) * cols:
            ax.set_xlabel('Cumulative Steps')

    plt.tight_layout()
    plot.save_or_show(fig_sub, filename='dkff_stability_flow_mag_separated.png')

    # 4.4 RMSE bar plot
    plot.plot_metrics_bar(
        rmse_dict,
        metric_name="RMSE",
        filename='dkff_rmse_bar.png'
    )


if __name__ == "__main__":
    main()