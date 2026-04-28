import copy
import os
import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf

from filterflow.base import State
from filterflow.models.simple_linear_gaussian import make_filter
from filterflow.resampling import RegularisedTransform
from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion
from utils.simple_linear_common import get_data

# If need eager debug, change to True
DEBUG_RUN_FUNCTIONS_EAGERLY = False
# Whether start tf.function
USE_TF_FUNCTION_FOR_ELBOS = False
VERBOSE = True


if DEBUG_RUN_FUNCTIONS_EAGERLY:
    tf.config.run_functions_eagerly(True)

def log(msg):
    if VERBOSE:
        print(msg, flush=True)

def timed_call(name, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    log(f"[TIMER] {name}: {dt:.3f} sec")
    return out, dt

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
    '''
    Set the threshold of neff.
    '''
    if resampling_neff == 0.0:
        return NeverResample()
    elif resampling_neff == 1.0:
        return AlwaysResample()
    else:
        return NeffCriterion(resampling_neff, True)


def make_elbo_runner(
    pf,
    initial_state,
    observations_dataset,
    T_tf,
    modifiable_transition_matrix,
    values_tf,
    filter_seed_tf,
    use_tf_function=False,
):
    """
    Calculate the log_likelihoods of pf
    """
    n_values = int(values_tf.shape[0])

    def runner_eager():
        elbos = []
        for i in range(n_values):
            val = values_tf[i]
            modifiable_transition_matrix.assign(tf.linalg.diag(val))
            final_state = pf(
                initial_state,
                observations_dataset,
                n_observations=T_tf,
                return_final=True,
                seed=filter_seed_tf
            )
            elbos.append(final_state.log_likelihoods / tf.cast(T_tf, tf.float32))
        return tf.stack(elbos, axis=0)

    if not use_tf_function:
        return runner_eager

    @tf.function(reduce_retracing=True)
    def runner_graph():
        elbos = tf.TensorArray(dtype=tf.float32, size=n_values)
        for i in tf.range(n_values):
            val = values_tf[i]
            assign_op = modifiable_transition_matrix.assign(tf.linalg.diag(val))
            with tf.control_dependencies([assign_op]):
                final_state = pf(
                    initial_state,
                    observations_dataset,
                    n_observations=T_tf,
                    return_final=True,
                    seed=filter_seed_tf
                )
                ll = final_state.log_likelihoods / tf.cast(T_tf, tf.float32)
            elbos = elbos.write(i, ll)
        return elbos.stack()
    return runner_graph


def summarize_pf_results(elbos,values,kf,data,T,epsilon,scaling,convergence_threshold,max_iter,n_particles,batch_size,runtime_sec):
    """
    output the results of 1/T(\hat{l}(\theta)-l(\theta))
    elbos: [n_theta, batch_size]
    """
    elbos_np = elbos.numpy()
    elbos_df = pd.DataFrame(elbos_np, index=pd.Index(values[:, 0], name="theta"))
    summary_df = elbos_df.T.describe().T[["mean", "std"]].reset_index()

    kalman_df = get_kalman_loglikelihoods(kf, data, values, T)

    out_df = summary_df.merge(kalman_df, on="theta", how="left")
    out_df["method"] = "OT"
    out_df["epsilon"] = epsilon
    out_df["scaling"] = scaling
    out_df["convergence_threshold"] = convergence_threshold
    out_df["max_iter"] = max_iter
    out_df["n_particles"] = n_particles
    out_df["batch_size"] = batch_size
    out_df["T"] = T
    out_df["runtime_sec"] = runtime_sec
    out_df["bias"] = out_df["mean"] - out_df["kalman_loglik"]
    out_df["abs_bias"] = np.abs(out_df["bias"])

    return out_df[
        [
            "method", "theta", "mean", "std",
            "kalman_loglik", "bias", "abs_bias",
            "epsilon", "scaling", "convergence_threshold", "max_iter",
            "n_particles", "batch_size", "T", "runtime_sec"
        ]
    ]

# Main experiment
def run_ot_experiment(
    epsilon,
    scaling,
    convergence_threshold,
    max_iter,
    T=150,
    batch_size=100,
    n_particles=25,
    data_seed=111,
    filter_seed=555,
    values=(0.25, 0.5, 0.75),
    resampling_neff=0.5,
    use_tf_function_for_elbos=USE_TF_FUNCTION_FOR_ELBOS,
):
    """
    Run entropy-regularized OT resampling and compare against Kalman baseline.
    """
    log(
        f"[START] epsilon={epsilon}, scaling={scaling}, "
        f"conv_thres={convergence_threshold}, max_iter={max_iter}, "
        f"T={T}, batch_size={batch_size}, n_particles={n_particles}"
    )

    # Model setup: Section 5.1 in Corenflos21a
    transition_matrix = 0.5 * np.eye(2, dtype=np.float32)
    transition_covariance = 0.5 * np.eye(2, dtype=np.float32)
    observation_matrix = np.eye(2, dtype=np.float32)
    observation_covariance = 0.1 * np.eye(2, dtype=np.float32)

    values = np.array(list(zip(values, values)), dtype=np.float32)

    # Generate data
    np_random_state = np.random.RandomState(seed=data_seed)
    data, kf = get_data(
        transition_matrix,
        observation_matrix,
        transition_covariance,
        observation_covariance,
        T,
        np_random_state
    )

    observations_dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(data, dtype=tf.float32)
    )

    # Resampling setup
    resampling_criterion = make_resampling_criterion(resampling_neff)
    resampling_method = RegularisedTransform(
        epsilon=epsilon,
        scaling=scaling,
        convergence_threshold=convergence_threshold,
        max_iter=max_iter,
    )

    # Modifiable transition matrix for changing theta
    init_transition_matrix = (
        0.5 * np.eye(2, dtype=np.float32)
        + 0.1 * np_random_state.randn(2, 2).astype(np.float32)
    )
    modifiable_transition_matrix = tf.Variable(
        init_transition_matrix,
        trainable=True,
        dtype=tf.float32
    )

    observation_matrix_tf = tf.convert_to_tensor(observation_matrix, dtype=tf.float32)
    transition_covariance_chol = tf.linalg.cholesky(
        tf.convert_to_tensor(transition_covariance, dtype=tf.float32)
    )
    observation_covariance_chol = tf.linalg.cholesky(
        tf.convert_to_tensor(observation_covariance, dtype=tf.float32)
    )

    # Initial particles / state
    initial_particles = np_random_state.normal(
        0.0, 1.0, [batch_size, n_particles, 2]
    ).astype(np.float32)

    initial_state = State(tf.constant(initial_particles, dtype=tf.float32))

    # Build SMC filter
    smc = make_filter(
        observation_matrix_tf,
        modifiable_transition_matrix,
        observation_covariance_chol,
        transition_covariance_chol,
        resampling_method,
        resampling_criterion
    )

    values_tf = tf.constant(values, dtype=tf.float32)
    T_tf = tf.constant(T, dtype=tf.int32)
    filter_seed_tf = tf.constant(filter_seed, dtype=tf.int32)

    # Build closure-based ELBO runner
    elbo_runner = make_elbo_runner(
        pf=smc,
        initial_state=initial_state,
        observations_dataset=observations_dataset,
        T_tf=T_tf,
        modifiable_transition_matrix=modifiable_transition_matrix,
        values_tf=values_tf,
        filter_seed_tf=filter_seed_tf,
        use_tf_function=use_tf_function_for_elbos,
    )

    try:
        log("[RUN] start")
        elbos, runtime_sec = timed_call("elbo_runner", elbo_runner)
        log("[RUN] done")
    except Exception as e:
        log("[RUN] failed")
        traceback.print_exc()
        raise RuntimeError(
            f"run_ot_experiment failed for "
            f"(epsilon={epsilon}, scaling={scaling}, "
            f"convergence_threshold={convergence_threshold}, "
            f"max_iter={max_iter}, n_particles={n_particles}) "
            f"with error: {e}"
        ) from e

    # Summarize
    out_df = summarize_pf_results(
        elbos=elbos,
        values=values,
        kf=kf,
        data=data,
        T=T,
        epsilon=epsilon,
        scaling=scaling,
        convergence_threshold=convergence_threshold,
        max_iter=max_iter,
        n_particles=n_particles,
        batch_size=batch_size,
        runtime_sec=runtime_sec,
    )

    log("[DONE] one experiment finished")
    return out_df

def run_all_ot_experiments(
    epsilons=(0.25, 0.5, 0.75),
    scalings=(0.9,),
    convergence_thresholds=(1e-2, 1e-3),
    max_iters=(20, 50),
    n_particles_list=(25,),
    T=150,
    batch_size=100,
    data_seed=111,
    filter_seed=555,
    values=(0.25, 0.5, 0.75),
    resampling_neff=0.5,
    save_csv=True,
    save_latex=False,
    out_dir="./tables",
    use_tf_function_for_elbos=USE_TF_FUNCTION_FOR_ELBOS,
):
    all_results = []

    total_jobs = (
        len(n_particles_list)
        * len(epsilons)
        * len(scalings)
        * len(convergence_thresholds)
        * len(max_iters)
    )
    job_id = 0

    for n_particles in n_particles_list:
        for epsilon in epsilons:
            for scaling in scalings:
                for convergence_threshold in convergence_thresholds:
                    for max_iter in max_iters:
                        job_id += 1
                        log("")
                        log("#" * 80)
                        log(f"[JOB {job_id}/{total_jobs}] start")

                        df = run_ot_experiment(
                            epsilon=epsilon,
                            scaling=scaling,
                            convergence_threshold=convergence_threshold,
                            max_iter=max_iter,
                            T=T,
                            batch_size=batch_size,
                            n_particles=n_particles,
                            data_seed=data_seed,
                            filter_seed=filter_seed,
                            values=values,
                            resampling_neff=resampling_neff,
                            use_tf_function_for_elbos=use_tf_function_for_elbos,
                        )
                        all_results.append(df)

                        log(f"[JOB {job_id}/{total_jobs}] done")

    result_df = pd.concat(all_results, ignore_index=True)
    result_df = result_df.sort_values(
        ["n_particles", "epsilon", "convergence_threshold", "max_iter", "theta"]
    ).reset_index(drop=True)

    print("\n=== OT resampling: per-theta results ===")
    print(result_df.to_string(index=False))

    summary_df = (
        result_df.groupby(
            [
                "method", "epsilon", "scaling", "convergence_threshold",
                "max_iter", "n_particles", "batch_size", "T"
            ],
            as_index=False
        )
        .agg(
            mean_bias=("bias", "mean"),
            mean_abs_bias=("abs_bias", "mean"),
            max_abs_bias=("abs_bias", "max"),
            mean_std=("std", "mean"),
            runtime_sec=("runtime_sec", "mean"),
        )
        .sort_values(["n_particles", "epsilon", "convergence_threshold", "max_iter"])
    )

    print("\n=== OT resampling: aggregated bias-variance-speed summary ===")
    print(summary_df.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)

    if save_csv:
        detail_csv = os.path.join(out_dir, f"ot_tradeoff_detail_T{T}_B{batch_size}.csv")
        summary_csv = os.path.join(out_dir, f"ot_tradeoff_summary_T{T}_B{batch_size}.csv")
        result_df.to_csv(detail_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        log(f"[SAVE] detail csv -> {detail_csv}")
        log(f"[SAVE] summary csv -> {summary_csv}")

    if save_latex:
        detail_tex = os.path.join(out_dir, f"ot_tradeoff_detail_T{T}_B{batch_size}.tex")
        summary_tex = os.path.join(out_dir, f"ot_tradeoff_summary_T{T}_B{batch_size}.tex")
        result_df.to_latex(detail_tex, float_format="%.4f", index=False)
        summary_df.to_latex(summary_tex, float_format="%.4f", index=False)
        log(f"[SAVE] detail tex -> {detail_tex}")
        log(f"[SAVE] summary tex -> {summary_tex}")

    return result_df, summary_df


if __name__ == "__main__":
    run_all_ot_experiments(
        epsilons=(0.25, 0.5, 0.75),
        scalings=(0.9,),
        convergence_thresholds=(1e-2, 1e-3),
        max_iters=(20, 50),
        n_particles_list=(25,),
        T=150,
        batch_size=100,
        data_seed=111,
        filter_seed=555,
        values=(0.25, 0.5, 0.75),
        resampling_neff=0.5,
        save_csv=True,
        save_latex=False,
        out_dir="./tables",
        use_tf_function_for_elbos=False,
    )