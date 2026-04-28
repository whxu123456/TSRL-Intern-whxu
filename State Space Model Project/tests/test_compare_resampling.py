import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.utils import filterflow_experiment_utils as utils
from src.utils import filterflow_diagnostics as diag
from filterflow.base import State

def test_ess_from_weights():
    weights = tf.constant([[[0.5, 0.5], [0.2, 0.8]]])  # shape [1,2,2]
    ess = diag.ess_from_weights(weights)
    expected = tf.constant([[2.0, 1.0 / 0.68]])
    tf.debugging.assert_near(ess, expected, rtol=1e-5)


def test_ess_ratio_from_weights():
    weights = tf.constant([[[0.3, 0.7], [0.6, 0.4]]])
    n = 2.0
    ess = diag.ess_from_weights(weights)
    ratio = diag.ess_ratio_from_weights(weights)
    expected = ess / n
    tf.debugging.assert_near(ratio, expected)


def test_weighted_particle_mean():
    weights = tf.constant([[[0.3, 0.7]]])
    particles = tf.constant([[[[1.0, 2.0], [3.0, 4.0]]]])
    mean = diag.weighted_particle_mean(weights, particles)
    expected = tf.constant([[[0.3*1+0.7*3, 0.3*2+0.7*4]]])
    tf.debugging.assert_near(mean, expected)


def test_weighted_cov_trace():
    weights = tf.constant([[[0.3, 0.7]]])
    particles = tf.constant([[[[0.0, 0.0], [2.0, 2.0]]]])
    trace = diag.weighted_cov_trace(weights, particles)
    tf.debugging.assert_near(trace, tf.constant([[[1.68]]]))


def test_avg_pairwise_sq_distance():
    particles = tf.constant([[[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]]])
    avg = diag.avg_pairwise_sq_distance(particles)
    expected = 32.0 / 9.0
    tf.debugging.assert_near(avg, tf.constant([[[expected]]]))


def test_particle_diversity():
    weights = tf.constant([[[0.5, 0.5]]])
    particles = tf.constant([[[[1.0, 1.0], [2.0, 2.0]]]])
    trace = diag.particle_diversity(weights, particles, mode="cov_trace")
    tf.debugging.assert_near(trace, tf.constant([[[0.5]]]))
    pair = diag.particle_diversity(weights, particles, mode="pairwise")
    tf.debugging.assert_near(pair, tf.constant([[[1.0]]]))
    with pytest.raises(ValueError):
        diag.particle_diversity(weights, particles, mode="invalid")


def test_infer_resampled_steps():
    n = 3
    log_weights_uniform = tf.fill([2, 2, n], -tf.math.log(tf.cast(n, tf.float32)))
    resampled = diag.infer_resampled_steps(log_weights_uniform)
    assert tf.reduce_all(resampled)
    log_weights_nonuni = tf.constant([[[-0.1, -100.0, -100.0]], [[-0.1, -100.0, -100.0]]])
    resampled2 = diag.infer_resampled_steps(log_weights_nonuni)
    assert not tf.reduce_any(resampled2)


def test_infer_soft_resampling_frequency():
    weights = tf.constant([
        [[0.6, 0.4], [0.9, 0.1]],
        [[0.5, 0.5], [0.7, 0.3]]
    ])
    threshold = 0.8
    flags = diag.infer_soft_resampling_frequency(weights, threshold)
    expected = tf.constant([[False, False], [False, True]])
    tf.debugging.assert_equal(flags, expected)


def test_summarize_states():
    class MockStateSeries:
        def __init__(self):
            self.particles = tf.random.normal([3, 2, 4, 2])
            self.weights = tf.nn.softmax(tf.random.normal([3, 2, 4]))
            self.log_weights = tf.math.log(self.weights)
            self.log_likelihoods = tf.random.normal([3, 2])
            self.resampling_flags = None
            self.ess_before_resampling = tf.random.uniform([3, 2])

    state = MockStateSeries()
    summary = diag.summarize_states(state, method="OT", resampling_neff=0.5, diversity_mode="cov_trace")
    expected_keys = [
        "loglik_per_batch", "mean_loglik", "std_loglik", "mean_ess_ratio", "min_ess_ratio",
        "resampling_frequency", "mean_diversity", "final_diversity",
        "mean_ess_before_resampling", "min_ess_before_resampling"
    ]
    for key in expected_keys:
        assert key in summary


# Tests for filterflow_experiment_utils.py
def test_make_observation_dataset():
    data = np.random.randn(10, 3).astype(np.float32)
    ds = utils.make_observation_dataset(data)
    for item in ds.take(1):
        assert item.shape == (3,)
        assert item.dtype == tf.float32


def test_make_dummy_inputs_dataset():
    T = 5
    ds = utils.make_dummy_inputs_dataset(T)
    items = list(ds.as_numpy_iterator())
    assert items == [0, 1, 2, 3, 4]


def test_make_initial_state():
    init_particles = np.random.randn(2, 10, 4).astype(np.float32)
    state = utils.make_initial_state(init_particles)
    assert isinstance(state, State)
    tf.debugging.assert_equal(state.particles, tf.convert_to_tensor(init_particles))


def test_build_filter_for_method():
    with patch("src.utils.filterflow_experiment_utils.make_filter") as mock_make_filter:
        mock_make_filter.return_value = "mock_filter"

        filt = utils.build_filter_for_method(
            method="NO",
            observation_matrix=tf.eye(2),
            transition_matrix=tf.eye(2),
            observation_error_chol=tf.eye(2),
            transition_noise_chol=tf.eye(2),
        )
        assert filt == "mock_filter"
        args, kwargs = mock_make_filter.call_args
        assert kwargs["resampling_method"].__class__.__name__ == "NoResampling"

        filt2 = utils.build_filter_for_method(
            method="SOFT",
            observation_matrix=tf.eye(2),
            transition_matrix=tf.eye(2),
            observation_error_chol=tf.eye(2),
            transition_noise_chol=tf.eye(2),
            alpha=0.2
        )
        args2, kwargs2 = mock_make_filter.call_args
        assert kwargs2["resampling_method"].__class__.__name__ == "SoftResampler"

        ot_kwargs = {"epsilon": 0.1, "scaling": 0.5, "convergence_threshold": 1e-2, "max_iter": 10}
        filt3 = utils.build_filter_for_method(
            method="OT",
            observation_matrix=tf.eye(2),
            transition_matrix=tf.eye(2),
            observation_error_chol=tf.eye(2),
            transition_noise_chol=tf.eye(2),
            ot_kwargs=ot_kwargs
        )
        args3, kwargs3 = mock_make_filter.call_args
        assert kwargs3["resampling_method"].epsilon == 0.1

        with pytest.raises(ValueError):
            utils.build_filter_for_method(method="UNKNOWN", observation_matrix=None, transition_matrix=None,
                                          observation_error_chol=None, transition_noise_chol=None)


def test__safe_get_state_field():
    class Dummy:
        foo = 42
    obj = Dummy()
    assert utils._safe_get_state_field(obj, "foo") == 42
    with pytest.raises(AttributeError):
        utils._safe_get_state_field(obj, "bar")


def test_replay_resampling_flags_from_states():
    # Create a mock smc that has a _resampling_criterion attribute with an apply method
    class MockCriterion:
        def apply(self, state):
            # Return tensors with batch dimension (batch size = 2 as per particles shape)
            return (tf.constant([True, False]), tf.constant([0.8, 0.9]))

    class MockSMC:
        def __init__(self):
            self._resampling_criterion = MockCriterion()

    smc = MockSMC()

    class MockSeries:
        def __init__(self):
            self.particles = tf.random.normal([3, 2, 4, 2])  # T=3, B=2
            self.weights = tf.random.uniform([3, 2, 4])
            self.log_weights = tf.math.log(self.weights)
            self.log_likelihoods = tf.random.normal([3, 2])
    states_series = MockSeries()
    initial_state = Mock()

    flags, ess = utils.replay_resampling_flags_from_states(smc, initial_state, states_series)
    assert flags.shape == (3, 2)
    assert ess.shape == (3, 2)
    # Verify the values are correctly propagated for each time step
    tf.debugging.assert_equal(flags[:, 0], tf.constant([True, True, True]))
    tf.debugging.assert_equal(flags[:, 1], tf.constant([False, False, False]))


def test_run_filterflow_smc():
    with patch("src.utils.filterflow_experiment_utils.make_initial_state") as mock_make_init, \
         patch("src.utils.filterflow_experiment_utils.make_observation_dataset") as mock_make_obs, \
         patch("src.utils.filterflow_experiment_utils.make_dummy_inputs_dataset") as mock_make_inputs, \
         patch("src.utils.filterflow_experiment_utils.replay_resampling_flags_from_states") as mock_replay:

        mock_state = Mock()
        mock_make_init.return_value = mock_state
        mock_obs_ds = Mock()
        mock_make_obs.return_value = mock_obs_ds
        mock_inputs_ds = Mock()
        mock_make_inputs.return_value = mock_inputs_ds

        mock_smc = Mock()
        mock_series = Mock()
        mock_series.particles = tf.random.normal([2, 3, 5, 2])
        mock_series.weights = tf.random.uniform([2, 3, 5])
        mock_series.log_weights = tf.math.log(mock_series.weights)
        mock_series.log_likelihoods = tf.random.normal([2, 3])
        mock_smc.return_value = mock_series

        mock_replay.return_value = (tf.ones([2, 3], dtype=tf.bool), tf.ones([2, 3]))

        observations = np.random.randn(2, 2).astype(np.float32)
        initial_particles = np.random.randn(3, 5, 2).astype(np.float32)

        result = utils.run_filterflow_smc(mock_smc, initial_particles, observations, seed=42)

        mock_smc.assert_called_once()
        args, kwargs = mock_smc.call_args
        assert kwargs["initial_state"] == mock_state
        assert kwargs["observation_series"] == mock_obs_ds
        assert kwargs["inputs_series"] == mock_inputs_ds
        assert kwargs["return_final"] is False
        assert kwargs["seed"].numpy().tolist() == [42, 43]

        assert isinstance(result, utils.TracedStateSeries)
        assert result.resampling_flags is not None
        assert result.ess_before_resampling is not None