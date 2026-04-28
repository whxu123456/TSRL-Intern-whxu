import copy
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.simple_linear_common import get_data
from utils.filterflow_experiment_utils import (
    build_filter_for_method,
    run_filterflow_smc,
)
from utils.filterflow_diagnostics import summarize_states


def get_kalman_loglikelihoods(kf, data, values, T):
    """
    Exact Kalman log-likelihood for each theta=(theta_1, theta_2)

    returns columns:
      theta
      kalman_loglik
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


def build_detail_df(
    method,
    values,
    diagnostics_list,
    kalman_df,
    runtime_sec,
    n_particles,
    batch_size,
    T,
    alpha=None,
    ot_kwargs=None,
):
    rows = []

    for i, theta in enumerate(values[:, 0]):
        diag = diagnostics_list[i]
        row = {
            "method": method,
            "theta": float(theta),
            "mean_loglik": diag["mean_loglik"],
            "std_loglik": diag["std_loglik"],
            "kalman_loglik": np.nan,
            "gap": np.nan,
            "abs_gap": np.nan,
            "mean_ess_ratio": diag["mean_ess_ratio"],
            "min_ess_ratio": diag["min_ess_ratio"],
            "resampling_frequency": diag["resampling_frequency"],
            "mean_diversity": diag["mean_diversity"],
            "final_diversity": diag["final_diversity"],
            "mean_ess_before_resampling": diag.get("mean_ess_before_resampling", np.nan),
            "min_ess_before_resampling": diag.get("min_ess_before_resampling", np.nan),
            "runtime_sec": float(runtime_sec),
            "n_particles": int(n_particles),
            "batch_size": int(batch_size),
            "T": int(T),
            "alpha": np.nan,
            "epsilon": np.nan,
            "scaling": np.nan,
            "convergence_threshold": np.nan,
            "max_iter": np.nan,
        }

        if method == "SOFT":
            row["alpha"] = alpha

        if method == "OT" and ot_kwargs is not None:
            row["epsilon"] = ot_kwargs.get("epsilon", np.nan)
            row["scaling"] = ot_kwargs.get("scaling", np.nan)
            row["convergence_threshold"] = ot_kwargs.get("convergence_threshold", np.nan)
            row["max_iter"] = ot_kwargs.get("max_iter", np.nan)

        rows.append(row)

    detail_df = pd.DataFrame(rows)
    detail_df = detail_df.drop(columns=["kalman_loglik", "gap", "abs_gap"]).merge(
        kalman_df, on="theta", how="left"
    )
    detail_df["gap"] = detail_df["mean_loglik"] - detail_df["kalman_loglik"]
    detail_df["abs_gap"] = detail_df["gap"].abs()

    ordered_cols = [
        "method", "theta",
        "mean_loglik", "std_loglik", "kalman_loglik", "gap", "abs_gap",
        "mean_ess_ratio", "min_ess_ratio",
        "mean_ess_before_resampling", "min_ess_before_resampling",
        "resampling_frequency",
        "mean_diversity", "final_diversity",
        "alpha", "epsilon", "scaling", "convergence_threshold", "max_iter",
        "n_particles", "batch_size", "T", "runtime_sec"
    ]
    return detail_df[ordered_cols]


def build_summary_df(detail_df):
    summary_df = (
        detail_df.groupby("method", as_index=False)
        .agg(
            mean_gap=("gap", "mean"),
            mean_abs_gap=("abs_gap", "mean"),
            max_abs_gap=("abs_gap", "max"),
            mean_std_loglik=("std_loglik", "mean"),
            mean_ess_ratio=("mean_ess_ratio", "mean"),
            min_ess_ratio=("min_ess_ratio", "mean"),
            mean_ess_before_resampling=("mean_ess_before_resampling", "mean"),
            min_ess_before_resampling=("min_ess_before_resampling", "mean"),
            mean_resampling_frequency=("resampling_frequency", "mean"),
            mean_diversity=("mean_diversity", "mean"),
            mean_final_diversity=("final_diversity", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return summary_df


def run_method(
    method,
    data,
    kf,
    values,
    T,
    batch_size,
    n_particles,
    data_seed,
    filter_seed,
    resampling_neff,
    alpha=None,
    ot_kwargs=None,
    diversity_mode="cov_trace",
):
    np_random_state = np.random.RandomState(seed=data_seed)

    initial_particles = np_random_state.normal(
        loc=0.0,
        scale=1.0,
        size=(batch_size, n_particles, 2)
    ).astype(np.float32)

    observation_matrix = np.eye(2, dtype=np.float32)
    observation_error_chol = np.linalg.cholesky(0.1 * np.eye(2, dtype=np.float32))
    transition_noise_chol = np.linalg.cholesky(0.5 * np.eye(2, dtype=np.float32))

    diagnostics_list = []

    t0 = time.perf_counter()

    for i, val in enumerate(values):
        transition_matrix = np.diag(val).astype(np.float32)

        smc = build_filter_for_method(
            method=method,
            observation_matrix=observation_matrix,
            transition_matrix=transition_matrix,
            observation_error_chol=observation_error_chol,
            transition_noise_chol=transition_noise_chol,
            resampling_neff=resampling_neff,
            alpha=alpha,
            ot_kwargs=ot_kwargs,
        )

        states_series = run_filterflow_smc(
            smc=smc,
            initial_particles=initial_particles,
            observations=data.astype(np.float32),
            seed=filter_seed + i,
        )

        diag = summarize_states(
            states_series=states_series,
            method=method,
            resampling_neff=resampling_neff,
            diversity_mode=diversity_mode,
        )
        diagnostics_list.append(diag)

    runtime_sec = time.perf_counter() - t0
    kalman_df = get_kalman_loglikelihoods(kf, data, values, T)

    return build_detail_df(
        method=method,
        values=values,
        diagnostics_list=diagnostics_list,
        kalman_df=kalman_df,
        runtime_sec=runtime_sec,
        n_particles=n_particles,
        batch_size=batch_size,
        T=T,
        alpha=alpha,
        ot_kwargs=ot_kwargs,
    )


def run_compare_experiment(
    T=150,
    batch_size=100,
    n_particles=25,
    data_seed=111,
    filter_seed=555,
    values=(0.25, 0.5, 0.75),
    resampling_neff=0.5,
    alpha=0.1,
    ot_kwargs=None,
    diversity_mode="cov_trace",
    save_csv=True,
    save_latex=False,
    out_dir="./tables",
):
    values = np.array(list(zip(values, values)), dtype=np.float32)

    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = 0.5 * np.eye(2, dtype=np.float32)
    observation_matrix = np.eye(2, dtype=np.float32)
    observation_covariance = 0.1 * np.eye(2, dtype=np.float32)

    np_random_state = np.random.RandomState(seed=data_seed)
    data, kf = get_data(
        transition_matrix,
        observation_matrix,
        transition_covariance,
        observation_covariance,
        T,
        np_random_state,
    )

    if ot_kwargs is None:
        ot_kwargs = {
            "epsilon": 0.5,
            "scaling": 0.9,
            "convergence_threshold": 1e-3,
            "max_iter": 50,
        }

    detail_no = run_method(
        method="NO",
        data=data,
        kf=kf,
        values=values,
        T=T,
        batch_size=batch_size,
        n_particles=n_particles,
        data_seed=data_seed,
        filter_seed=filter_seed,
        resampling_neff=resampling_neff,
        diversity_mode=diversity_mode,
    )

    detail_soft = run_method(
        method="SOFT",
        data=data,
        kf=kf,
        values=values,
        T=T,
        batch_size=batch_size,
        n_particles=n_particles,
        data_seed=data_seed,
        filter_seed=filter_seed,
        resampling_neff=resampling_neff,
        alpha=alpha,
        ot_kwargs=None,
        diversity_mode=diversity_mode,
    )

    detail_ot = run_method(
        method="OT",
        data=data,
        kf=kf,
        values=values,
        T=T,
        batch_size=batch_size,
        n_particles=n_particles,
        data_seed=data_seed,
        filter_seed=filter_seed,
        resampling_neff=resampling_neff,
        alpha=None,
        ot_kwargs=ot_kwargs,
        diversity_mode=diversity_mode,
    )

    detail_df = pd.concat([detail_no, detail_soft, detail_ot], ignore_index=True)
    summary_df = build_summary_df(detail_df)

    print("\n=== Detailed comparison table ===")
    print(detail_df.to_string(index=False))

    print("\n=== Summary table ===")
    print(summary_df.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)

    if save_csv:
        detail_csv = os.path.join(
            out_dir,
            f"compare_resampling_detail_T{T}_N{n_particles}_B{batch_size}.csv"
        )
        summary_csv = os.path.join(
            out_dir,
            f"compare_resampling_summary_T{T}_N{n_particles}_B{batch_size}.csv"
        )
        detail_df.to_csv(detail_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nSaved detail csv to: {detail_csv}")
        print(f"Saved summary csv to: {summary_csv}")

    if save_latex:
        detail_tex = os.path.join(
            out_dir,
            f"compare_resampling_detail_T{T}_N{n_particles}_B{batch_size}.tex"
        )
        summary_tex = os.path.join(
            out_dir,
            f"compare_resampling_summary_T{T}_N{n_particles}_B{batch_size}.tex"
        )
        detail_df.to_latex(detail_tex, float_format="%.4f", index=False)
        summary_df.to_latex(summary_tex, float_format="%.4f", index=False)
        print(f"Saved detail tex to: {detail_tex}")
        print(f"Saved summary tex to: {summary_tex}")

    return detail_df, summary_df


if __name__ == "__main__":
    tf.random.set_seed(123)

    run_compare_experiment(
        T=150,
        batch_size=100,
        n_particles=25,
        data_seed=111,
        filter_seed=555,
        values=(0.25, 0.5, 0.75),
        resampling_neff=0.5,
        alpha=0.1,
        ot_kwargs={
            "epsilon": 0.5,
            "scaling": 0.9,
            "convergence_threshold": 1e-3,
            "max_iter": 50,
        },
        diversity_mode="cov_trace",
        save_csv=True,
        save_latex=False,
        out_dir="./tables",
    )