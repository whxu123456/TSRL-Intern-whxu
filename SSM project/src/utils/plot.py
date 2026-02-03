import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2


def save_or_show(fig, filename=None, output_dir='results'):
    if filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Picture saved: {path}")
    else:
        plt.show()
    plt.close(fig)


def plot_trajectory_comparison(x_true, estimates_dict, title="State Estimation Comparison", filename=None):
    """
    Compare the trajectory of true states and the estimated stated by different algorithms
    estimates_dict: {'EKF': x_ekf, 'UKF': x_ukf, ...}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_true, 'k-', label='True State', alpha=0.6, linewidth=2.5)

    colors = sns.color_palette("husl", len(estimates_dict))
    for i, (name, est) in enumerate(estimates_dict.items()):
        ax.plot(est, label=name, linestyle='--', color=colors[i], alpha=0.8)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("State Value")
    ax.legend()
    save_or_show(fig, filename)


def plot_error_metrics(errors_dict, title="Estimation Error over Time", filename=None):
    """
    Plot the estimation error over time
    """

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, error in errors_dict.items():
        ax.plot(error, label=f'{name} Error')

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Absolute Error / Norm")
    ax.set_yscale('log')
    ax.legend()
    save_or_show(fig, filename)

def plot_condition_number(errors_dict, title="Condition Number Comparison", filename=None):
    """
    Plot the Condition Number over time
    """

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, error in errors_dict.items():
        ax.plot(error, label=f'{name}')

    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Condition Number")
    ax.set_yscale('log')
    ax.legend()
    save_or_show(fig, filename)

def plot_metrics_bar(metrics_data, metric_name="RMSE", filename=None):
    """
    plot the bar plot of RMSE
    metrics_data: {'EKF': 0.5, 'UKF': 0.4, ...}
    """
    names = list(metrics_data.keys())
    values = list(metrics_data.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=names, y=values, ax=ax, palette="viridis")
    ax.set_title(f"{metric_name} Comparison")
    ax.set_ylabel(metric_name)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')

    save_or_show(fig, filename)


def plot_acoustic_tracking(sensors_pos, x_true, est_traj_dict, n_targets, filename=None):
    """
    2-d trajectory picture in the Acoustic tracking problem (Li(17))
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    #draw sensors
    ax.scatter(sensors_pos[:, 0], sensors_pos[:, 1], c='k', marker='s', s=50, label='Sensors', alpha=0.5)

    colors = ['b', 'r', 'g', 'm', 'c']

    # Draw the true trajectory
    for i in range(n_targets):
        tx = x_true[:, i * 4]
        ty = x_true[:, i * 4 + 1]
        ax.plot(tx, ty, color=colors[i % len(colors)], linestyle='-', linewidth=2, label=f'Target {i + 1} True')
        ax.plot(tx[0], ty[0], marker='x', color=colors[i % len(colors)], markersize=10)  # 起点

    # Draw the estimated trajectory
    linestyles = ['--', ':', '-.']
    for alg_idx, (alg_name, traj) in enumerate(est_traj_dict.items()):
        style = linestyles[alg_idx % len(linestyles)]
        for i in range(n_targets):
            ex = traj[:, i * 4]
            ey = traj[:, i * 4 + 1]
            label = f'{alg_name}' if i == 0 else None
            ax.plot(ex, ey, color=colors[i % len(colors)], linestyle=style, linewidth=1.5, alpha=0.7, label=label)

    ax.set_title('Multi-Target Tracking Trajectories')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    ax.axis('equal')
    save_or_show(fig, filename)


def plot_lorenz_collapse(prior, posterior, truth, obs_indices, title_suffix="", filename=None):
    """
    Plot Lorenz96 (Hu(21))
    """
    idx_obs = obs_indices[0]
    idx_unobs = idx_obs - 1 if idx_obs > 0 else idx_obs + 1

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(prior[:, idx_unobs], prior[:, idx_obs], c='gray', alpha=0.4, label='Prior', s=30)
    ax.scatter(posterior[:, idx_unobs], posterior[:, idx_obs], c='red', alpha=0.8, label='Posterior', s=30)
    ax.scatter(truth[idx_unobs], truth[idx_obs], c='blue', marker='*', s=200, label='Truth')

    ax.set_title(f"Particle Distribution: {title_suffix}\nUnobserved x({idx_unobs}) vs Observed x({idx_obs})")
    ax.set_xlabel(f"Unobserved State x({idx_unobs})")
    ax.set_ylabel(f"Observed State x({idx_obs})")
    ax.legend()
    save_or_show(fig, filename)