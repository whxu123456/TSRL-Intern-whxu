import tensorflow as tf

from filterflow.base import State
from filterflow.models.simple_linear_gaussian import make_filter
from filters.Diffpf_soft_resampling import SoftResampler
from filterflow.resampling.criterion import NeffCriterion, NeverResample
from filterflow.resampling import NoResampling
from filterflow.resampling import RegularisedTransform

class TracedStateSeries:
    """
    Lightweight container for diagnostics.

    fields:
      particles              [T, B, N, dx]
      weights                [T, B, N]
      log_weights            [T, B, N]
      log_likelihoods        [T, B]
      resampling_flags       [T, B]     bool
      ess_before_resampling  [T, B]     float or None
    """
    def __init__(
        self,
        particles,
        weights,
        log_weights,
        log_likelihoods,
        resampling_flags=None,
        ess_before_resampling=None,
    ):
        self.particles = particles
        self.weights = weights
        self.log_weights = log_weights
        self.log_likelihoods = log_likelihoods
        self.resampling_flags = resampling_flags
        self.ess_before_resampling = ess_before_resampling


def make_observation_dataset(data):
    """
    data: [T, dx]
    """
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    return tf.data.Dataset.from_tensor_slices(data)


def make_dummy_inputs_dataset(T):
    """
    Explicitly provide equal-length inputs dataset to avoid inconsistent behavior in graph mode.
    """
    return tf.data.Dataset.range(tf.cast(T, tf.int64), output_type=tf.int32)


def make_initial_state(initial_particles):
    """
    initial_particles: [B, N, dx]
    """
    initial_particles = tf.convert_to_tensor(initial_particles, dtype=tf.float32)
    return State(particles=initial_particles)


def build_filter_for_method(
    method,
    observation_matrix,
    transition_matrix,
    observation_error_chol,
    transition_noise_chol,
    resampling_neff=0.5,
    alpha=0.1,
    ot_kwargs=None,
):
    method = method.upper()

    if method == "NO":
        resampling_method = NoResampling()
        resampling_criterion = NeverResample()

    elif method == "SOFT":
        resampling_method = SoftResampler(alpha=alpha, on_log=True)
        resampling_criterion = NeffCriterion(
            threshold=resampling_neff,
            is_relative=True,
            on_log=True,
            assume_normalized=True,
        )

    elif method == "OT":
        if ot_kwargs is None:
            ot_kwargs = {
                "epsilon": 0.5,
                "scaling": 0.9,
                "convergence_threshold": 1e-3,
                "max_iter": 50,
            }

        resampling_method = RegularisedTransform(
            epsilon=ot_kwargs["epsilon"],
            scaling=ot_kwargs["scaling"],
            convergence_threshold=ot_kwargs["convergence_threshold"],
            max_iter=ot_kwargs["max_iter"],
        )
        resampling_criterion = NeffCriterion(
            threshold=resampling_neff,
            is_relative=True,
            on_log=True,
            assume_normalized=True,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    smc = make_filter(
        observation_matrix=observation_matrix,
        transition_matrix=transition_matrix,
        observation_error_chol=observation_error_chol,
        transition_noise_chol=transition_noise_chol,
        resampling_method=resampling_method,
        resampling_criterion=resampling_criterion,
        optimal_proposal=False,
    )
    return smc


def _safe_get_state_field(state_like, name):
    if hasattr(state_like, name):
        return getattr(state_like, name)
    raise AttributeError(f"State-like object has no attribute '{name}'")


def replay_resampling_flags_from_states(smc, initial_state, states_series):
    """
    Replay filterflow's own resampling criterion step by step on the "pre-update" states
    to obtain strictly defined resampling flags.

    pre-update state 序列为：
      t=0: initial_state
      t>=1: states_series[t-1]

    returns:
      resampling_flags      [T, B] bool
      ess_before_resampling [T, B] float
    """
    particles = tf.convert_to_tensor(_safe_get_state_field(states_series, "particles"))
    weights = tf.convert_to_tensor(_safe_get_state_field(states_series, "weights"))
    log_weights = tf.convert_to_tensor(_safe_get_state_field(states_series, "log_weights"))
    log_likelihoods = tf.convert_to_tensor(_safe_get_state_field(states_series, "log_likelihoods"))

    T = int(particles.shape[0])

    flag_hist = []
    ess_hist = []

    for t in range(T):
        if t == 0:
            pre_state = initial_state
        else:
            pre_state = State(
                particles=particles[t - 1],
                weights=weights[t - 1],
                log_weights=log_weights[t - 1],
                log_likelihoods=log_likelihoods[t - 1],
            )

        out = smc._resampling_criterion.apply(pre_state)

        if isinstance(out, tuple) and len(out) == 2:
            flag_t, ess_t = out
        else:
            raise ValueError(
                "Expected smc._resampling_criterion.apply(state) "
                "to return (flag, ess), but got incompatible output."
            )

        flag_hist.append(tf.convert_to_tensor(flag_t, dtype=tf.bool))
        ess_hist.append(tf.convert_to_tensor(ess_t, dtype=tf.float32))

    resampling_flags = tf.stack(flag_hist, axis=0)          # [T, B]
    ess_before_resampling = tf.stack(ess_hist, axis=0)      # [T, B]

    return resampling_flags, ess_before_resampling


def run_filterflow_smc(
    smc,
    initial_particles,
    observations,
    seed=123,
):
    """
    Run original filterflow SMC, then replay exact resampling criterion on the
    pre-update states to obtain strict resampling flags.

    returns:
        TracedStateSeries with fields:
            particles              [T, B, N, dx]
            weights                [T, B, N]
            log_weights            [T, B, N]
            log_likelihoods        [T, B]
            resampling_flags       [T, B]
            ess_before_resampling  [T, B]
    """
    observations = tf.convert_to_tensor(observations, dtype=tf.float32)
    T = int(observations.shape[0])

    initial_state = make_initial_state(initial_particles)
    obs_ds = make_observation_dataset(observations)
    inputs_ds = make_dummy_inputs_dataset(T)

    states_series = smc(
        initial_state=initial_state,
        observation_series=obs_ds,
        n_observations=tf.constant(T, dtype=tf.int32),
        inputs_series=inputs_ds,
        return_final=False,
        seed=tf.constant([seed, seed + 1], dtype=tf.int32),
    )

    resampling_flags, ess_before_resampling = replay_resampling_flags_from_states(
        smc=smc,
        initial_state=initial_state,
        states_series=states_series,
    )

    return TracedStateSeries(
        particles=tf.convert_to_tensor(states_series.particles),
        weights=tf.convert_to_tensor(states_series.weights),
        log_weights=tf.convert_to_tensor(states_series.log_weights),
        log_likelihoods=tf.convert_to_tensor(states_series.log_likelihoods),
        resampling_flags=resampling_flags,
        ess_before_resampling=ess_before_resampling,
    )