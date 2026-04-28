import copy
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from main_Diffpf_LGSSM import (
    make_resampling_criterion,
    timed_call,
    get_kalman_loglikelihoods,
    summarize_pf_results,
    make_elbo_runner,
    run_ot_experiment,
    run_all_ot_experiments,
)

tf.config.experimental.set_visible_devices([], "GPU")


def test_make_resampling_criterion():
    from filterflow.resampling.criterion import NeverResample, AlwaysResample, NeffCriterion

    assert isinstance(make_resampling_criterion(0.0), NeverResample)
    assert isinstance(make_resampling_criterion(1.0), AlwaysResample)
    crit = make_resampling_criterion(0.5)
    assert isinstance(crit, NeffCriterion)

def test_timed_call():
    def dummy_func(x):
        return x * 2

    result, duration = timed_call("dummy", dummy_func, 21)
    assert result == 42
    assert duration > 0


def test_get_kalman_loglikelihoods(monkeypatch):
    # Mock copy.copy to return a Kalman filter that always returns a fixed log-likelihood
    class MockKF:
        transition_matrices = None

        def loglikelihood(self, data):
            return 42.0

    def mock_copy(kf_orig):
        return MockKF()

    monkeypatch.setattr(copy, "copy", mock_copy)

    data = np.zeros((3, 2))  # T=3
    values = np.array([[0.2, 0.2], [0.5, 0.5]], dtype=np.float32)
    T = 3

    df = get_kalman_loglikelihoods(None, data, values, T)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["theta", "kalman_loglik"]
    assert len(df) == 2
    # Each loglikelihood is 42.0 divided by T
    assert (df["kalman_loglik"] == 42.0 / T).all()


def test_summarize_pf_results():
    elbos = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=tf.float32)
    values = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=np.float32)

    from unittest.mock import patch
    with patch("main_Diffpf_LGSSM.get_kalman_loglikelihoods") as mock_kalman:
        mock_kalman.return_value = pd.DataFrame({
            "theta": [0.25, 0.75],
            "kalman_loglik": [0.15, 0.55]
        })
        df = summarize_pf_results(
            elbos=elbos,
            values=values,
            kf=None,
            data=None,
            T=10,
            epsilon=0.5,
            scaling=0.9,
            convergence_threshold=1e-3,
            max_iter=20,
            n_particles=25,
            batch_size=100,
            runtime_sec=1.23,
        )

    expected_columns = [
        "method", "theta", "mean", "std", "kalman_loglik", "bias", "abs_bias",
        "epsilon", "scaling", "convergence_threshold", "max_iter",
        "n_particles", "batch_size", "T", "runtime_sec"
    ]
    assert list(df.columns) == expected_columns
    assert len(df) == 2
    # Check computed means
    assert df["mean"].iloc[0] == pytest.approx(0.2)   # (0.1+0.2+0.3)/3
    assert df["mean"].iloc[1] == pytest.approx(0.5)   # (0.4+0.5+0.6)/3


def test_make_elbo_runner():
    # Minimal dummy Particle Filter
    class DummyPF:
        def __call__(self, initial_state, observations_dataset, n_observations, return_final, seed):
            class DummyState:
                log_likelihoods = tf.constant(5.0)
            return DummyState()

    pf = DummyPF()
    initial_state = tf.constant(0.0)
    observations_dataset = tf.data.Dataset.from_tensor_slices([1.0, 2.0])
    T_tf = tf.constant(2)
    modifiable_transition_matrix = tf.Variable([[0.5, 0.0], [0.0, 0.5]], dtype=tf.float32)
    values_tf = tf.constant([[0.2, 0.2], [0.8, 0.8]], dtype=tf.float32)
    filter_seed_tf = tf.constant(123)

    runner_eager = make_elbo_runner(
        pf, initial_state, observations_dataset, T_tf,
        modifiable_transition_matrix, values_tf,
        filter_seed_tf, use_tf_function=False
    )
    assert callable(runner_eager)
    result = runner_eager()
    assert isinstance(result, tf.Tensor)
    assert result.shape[0] == values_tf.shape[0]

    runner_graph = make_elbo_runner(
        pf, initial_state, observations_dataset, T_tf,
        modifiable_transition_matrix, values_tf,
        filter_seed_tf, use_tf_function=True
    )
    assert callable(runner_graph)
    result2 = runner_graph()
    assert isinstance(result2, tf.Tensor)


def test_run_ot_experiment():
    df = run_ot_experiment(
        epsilon=0.5,
        scaling=0.9,
        convergence_threshold=0.1,
        max_iter=5,
        T=3,
        batch_size=2,
        n_particles=3,
        data_seed=111,
        filter_seed=555,
        values=(0.3, 0.7),
        resampling_neff=0.5,
        use_tf_function_for_elbos=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    expected_cols = [
        "method", "theta", "mean", "std", "kalman_loglik", "bias", "abs_bias",
        "epsilon", "scaling", "convergence_threshold", "max_iter",
        "n_particles", "batch_size", "T", "runtime_sec"
    ]
    assert all(col in df.columns for col in expected_cols)
    assert df["mean"].notnull().all()
    assert df["std"].notnull().all()

def test_run_all_ot_experiments(tmp_path):
    out_dir = tmp_path / "test_tables"
    result_df, summary_df = run_all_ot_experiments(
        epsilons=(0.5,),
        scalings=(0.9,),
        convergence_thresholds=(0.1,),
        max_iters=(5,),
        n_particles_list=(3,),
        T=3,
        batch_size=2,
        data_seed=111,
        filter_seed=555,
        values=(0.3, 0.7),
        resampling_neff=0.5,
        save_csv=True,
        save_latex=False,
        out_dir=str(out_dir),
        use_tf_function_for_elbos=False,
    )
    assert isinstance(result_df, pd.DataFrame)
    assert isinstance(summary_df, pd.DataFrame)
    assert len(result_df) == 2
    assert len(summary_df) == 1
    assert (out_dir / "ot_tradeoff_detail_T3_B2.csv").exists()
    assert (out_dir / "ot_tradeoff_summary_T3_B2.csv").exists()