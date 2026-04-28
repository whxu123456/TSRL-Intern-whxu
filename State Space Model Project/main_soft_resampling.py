import copy
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from filterflow.base import State
from filterflow.models.simple_linear_gaussian import make_filter
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from utils.simple_linear_common import get_data
from filters.Diffpf_soft_resampling import SoftResampler


@tf.function(experimental_relax_shapes=True)
def get_elbos(pf, initial_state, observations_dataset, T, modifiable_transition_values, values, filter_seed):
    """
    Compute log-likelihood estimates / T for each theta value in `values`.
    Returns shape: [n_values, batch_size]
    """
    elbos = tf.TensorArray(dtype=tf.float32, size=values.shape[0])

    for i in tf.range(values.shape[0]):
        val = values[i]
        assign_op = modifiable_transition_values.assign(tf.linalg.diag(val))
        with tf.control_dependencies([assign_op]):
            final_state = pf(
                initial_state,
                observations_dataset,
                n_observations=T,
                return_final=True,
                seed=filter_seed
            )
        elbos = elbos.write(
            tf.cast(i, tf.int32),
            final_state.log_likelihoods / tf.cast(T, tf.float32)
        )

    return elbos.stack()


def get_kalman_loglikelihoods(kf, data, values, T):
    """
    Exact Kalman log-likelihood / T for each theta=(theta_1, theta_2).
    """
    log_likelihoods = []
    for val in values:
        transition_matrix = np.diag(val)
        kf_copy = copy.copy(kf)
        kf_copy.transition_matrices = transition_matrix
        log_likelihoods.append(kf_copy.loglikelihood(data) / T)

    return pd.DataFrame({
        "theta": values[:, 0],
        "kalman_loglik": log_likelihoods,
    })


def make_resampling_criterion(resampling_neff):
    if resampling_neff == 0.0:
        return NeverResample()
    elif resampling_neff == 1.0:
        return AlwaysResample()
    else:
        return NeffCriterion(resampling_neff, True)


def run_soft_resampling_experiment(
    alpha,
    T=150,
    batch_size=100,
    n_particles=25,
    data_seed=111,
    filter_seed=555,
    values=(0.25, 0.5, 0.75),
    resampling_neff=0.5,
):
    """
    Run soft-resampling PF and compare with Kalman exact log-likelihood.
    """
    # Model setup for Section 5.1
    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = np.eye(2, dtype=np.float32)
    observation_matrix = np.eye(2, dtype=np.float32)
    observation_covariance = 0.1 * np.eye(2, dtype=np.float32)

    values = np.array(list(zip(values, values)), dtype=np.float32)

    # Generate shared dataset + Kalman baseline
    np_random_state = np.random.RandomState(seed=data_seed)
    data, kf = get_data(
        transition_matrix,
        observation_matrix,
        transition_covariance,
        observation_covariance,
        T,
        np_random_state
    )
    observation_dataset = tf.data.Dataset.from_tensor_slices(data)

    # Resampling
    resampling_criterion = make_resampling_criterion(resampling_neff)
    resampling_method = SoftResampler(alpha=alpha, on_log=True)

    # Trainable transition matrix overwritten inside get_elbos
    init_transition_matrix = (
        0.5 * np.eye(2) + 0.1 * np_random_state.randn(2, 2)
    ).astype(np.float32)
    modifiable_transition_matrix = tf.Variable(init_transition_matrix, trainable=True)

    observation_matrix_tf = tf.convert_to_tensor(observation_matrix)
    transition_covariance_chol = tf.linalg.cholesky(transition_covariance)
    observation_covariance_chol = tf.linalg.cholesky(observation_covariance)

    initial_particles = np_random_state.normal(
        0.0, 1.0, [batch_size, n_particles, 2]
    ).astype(np.float32)
    initial_state = State(tf.constant(initial_particles))

    smc = make_filter(
        observation_matrix_tf,
        modifiable_transition_matrix,
        observation_covariance_chol,
        transition_covariance_chol,
        resampling_method,
        resampling_criterion
    )

    start_time = time.perf_counter()
    elbos = get_elbos(
        smc,
        initial_state,
        observation_dataset,
        tf.constant(T),
        modifiable_transition_matrix,
        tf.constant(values),
        tf.constant(filter_seed)
    )
    runtime_sec = time.perf_counter() - start_time

    # PF summary
    elbos_df = pd.DataFrame(elbos.numpy(), index=pd.Index(values[:, 0], name="theta"))
    summary_df = elbos_df.T.describe().T[["mean", "std"]].reset_index()

    # Kalman baseline
    kalman_df = get_kalman_loglikelihoods(kf, data, values, T)

    # Merge + gap stats
    out_df = summary_df.merge(kalman_df, on="theta", how="left")
    out_df["method"] = "SOFT"
    out_df["alpha"] = alpha
    out_df["n_particles"] = n_particles
    out_df["batch_size"] = batch_size
    out_df["T"] = T
    out_df["runtime_sec"] = runtime_sec
    out_df["gap"] = out_df["mean"] - out_df["kalman_loglik"]
    out_df["abs_gap"] = np.abs(out_df["gap"])

    return out_df[
        [
            "method", "alpha", "theta", "mean", "std",
            "kalman_loglik", "gap", "abs_gap",
            "n_particles", "batch_size", "T", "runtime_sec"
        ]
    ]


def run_all_soft_experiments(
    alphas=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
    T=150,
    batch_size=100,
    n_particles=25,
    data_seed=111,
    filter_seed=555,
    values=(0.25, 0.5, 0.75),
    resampling_neff=0.5,
    save_csv=True,
    save_latex=False,
    out_dir="./tables",
):
    all_results = []

    for alpha in alphas:
        df = run_soft_resampling_experiment(
            alpha=alpha,
            T=T,
            batch_size=batch_size,
            n_particles=n_particles,
            data_seed=data_seed,
            filter_seed=filter_seed,
            values=values,
            resampling_neff=resampling_neff,
        )
        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)
    result_df = result_df.sort_values(["alpha", "theta"]).reset_index(drop=True)

    print("\n Soft-resampling vs Kalman: per-theta results")
    print(result_df.to_string(index=False))

    summary_df = (
        result_df.groupby(["method", "alpha", "n_particles", "batch_size", "T"], as_index=False)
        .agg(
            mean_gap=("gap", "mean"),
            mean_abs_gap=("abs_gap", "mean"),
            max_abs_gap=("abs_gap", "max"),
            mean_std=("std", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(["alpha"])
    )

    print("Soft-resampling vs Kalman: aggregated summary")
    print(summary_df.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)

    if save_csv:
        result_df.to_csv(
            os.path.join(out_dir, f"soft_vs_kalman_detail_T{T}_N{n_particles}_B{batch_size}.csv"),
            index=False
        )
        summary_df.to_csv(
            os.path.join(out_dir, f"soft_vs_kalman_summary_T{T}_N{n_particles}_B{batch_size}.csv"),
            index=False
        )

    if save_latex:
        result_df.to_latex(
            os.path.join(out_dir, f"soft_vs_kalman_detail_T{T}_N{n_particles}_B{batch_size}.tex"),
            float_format="%.4f",
            index=False
        )
        summary_df.to_latex(
            os.path.join(out_dir, f"soft_vs_kalman_summary_T{T}_N{n_particles}_B{batch_size}.tex"),
            float_format="%.4f",
            index=False
        )

    return result_df, summary_df


if __name__ == "__main__":
    run_all_soft_experiments(
        alphas=(0.0, 0.1, 0.3, 0.5, 0.7, 1.0),
        T=150,
        batch_size=100,
        n_particles=25,
        data_seed=111,
        filter_seed=555,
        values=(0.25, 0.5, 0.75),
        resampling_neff=0.5,
        save_csv=True,
        save_latex=False,
        out_dir="./tables",
    )