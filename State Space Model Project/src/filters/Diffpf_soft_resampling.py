import tensorflow as tf
import attr

from filterflow.base import State
from filterflow.resampling.base import resample
from filterflow.resampling.standard.base import (
    StandardResamplerBase,
    _discrete_percentile_function,
)
from filterflow.resampling.standard.multinomial import _uniform_spacings

''' 
Soft-resampling method adapted from the standard particle filter method
'''

@tf.function
def _normalize_weights(weights, eps=1e-12):
    weights = tf.maximum(weights, eps)
    weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
    return weights


@tf.function
def _weights_to_log_weights(weights, eps=1e-12):
    weights = tf.maximum(weights, eps)
    log_weights = tf.math.log(weights)
    log_weights = log_weights - tf.reduce_logsumexp(log_weights, axis=1, keepdims=True)
    return log_weights


@tf.function
def _log_weights_to_weights(log_weights):
    weights = tf.nn.softmax(log_weights, axis=1)
    return weights


class SoftResampler(StandardResamplerBase):
    """
    Soft-resampling: q(i) = alpha * w(i) + (1 - alpha) / N
    Then sample ancestors from q, and assign importance-corrected weights:
        w_new(j) ∝ w_old(a_j) / q(a_j)
    """
    DIFFERENTIABLE = True

    def __init__(self, alpha=0.5, on_log=True, eps=1e-12, name='SoftResampler'):
        """
        alpha: float in [0, 1]
            Mixture coefficient. alpha=1 -> standard resampling;
            alpha=0 -> uniform ancestor sampling.
        param on_log: Whether to use log-weights in percentile function.
        """
        super(SoftResampler, self).__init__(name=name, on_log=on_log)
        self._alpha = alpha
        self._eps = eps

    @staticmethod
    def _get_spacings(n_particles, batch_size, seed):
        # Reuse multinomial spacings
        return _uniform_spacings(n_particles, batch_size, seed)

    def apply(self, state: State, flags: tf.Tensor, seed=None):
        """
        Resample according to q = alpha * w + (1-alpha)/N,
        then apply importance correction w/q to new weights.
        """
        batch_size = state.batch_size
        n_particles = state.n_particles

        float_n_particles = tf.cast(n_particles, state.weights.dtype)
        uniform_weights = tf.ones_like(state.weights) / float_n_particles

        # Convert to ordinary weights if needed
        if self._on_log:
            old_weights = _log_weights_to_weights(state.log_weights)
        else:
            old_weights = _normalize_weights(state.weights, self._eps)

        # Mixture distribution q
        alpha = tf.cast(self._alpha, old_weights.dtype)
        proposal_weights = alpha * old_weights + (1.0 - alpha) * uniform_weights
        proposal_weights = _normalize_weights(proposal_weights, self._eps)
        proposal_log_weights = _weights_to_log_weights(proposal_weights, self._eps)

        # Draw ancestor indices using proposal q
        spacings = self._get_spacings(n_particles, batch_size, seed)
        indices = _discrete_percentile_function(
            spacings,
            n_particles,
            self._on_log,
            proposal_weights,
            proposal_log_weights
        )

        ancestor_indices = tf.where(
            tf.reshape(flags, [-1, 1]),
            indices,
            tf.reshape(tf.range(n_particles), [1, -1])
        )

        # Gather resampled particles using sampled ancestors
        new_particles = tf.gather(
            state.particles, indices, axis=1, batch_dims=1, validate_indices=False
        )

        # Importance correction:
        # new_w_j ∝ old_w[a_j] / proposal_w[a_j]
        gathered_old_weights = tf.gather(
            old_weights, indices, axis=1, batch_dims=1, validate_indices=False
        )
        gathered_proposal_weights = tf.gather(
            proposal_weights, indices, axis=1, batch_dims=1, validate_indices=False
        )

        corrected_weights = gathered_old_weights / tf.maximum(gathered_proposal_weights, self._eps)
        corrected_weights = _normalize_weights(corrected_weights, self._eps)
        corrected_log_weights = _weights_to_log_weights(corrected_weights, self._eps)

        # Respect flags: if no resampling, keep original state
        if self._on_log:
            original_weights = old_weights
            original_log_weights = state.log_weights
        else:
            original_weights = _normalize_weights(state.weights, self._eps)
            original_log_weights = _weights_to_log_weights(original_weights, self._eps)

        resampled_particles = resample(state.particles, new_particles, flags)
        resampled_weights = resample(original_weights, corrected_weights, flags)
        resampled_log_weights = resample(original_log_weights, corrected_log_weights, flags)

        additional_variables = {}
        for additional_state_variable in state.ADDITIONAL_STATE_VARIABLES:
            state_variable = getattr(state, additional_state_variable)
            new_state_variable = tf.gather(
                state_variable, indices, axis=1, batch_dims=1, validate_indices=False
            )
            additional_variables[additional_state_variable] = resample(
                state_variable, new_state_variable, flags
            )

        return attr.evolve(
            state,
            particles=resampled_particles,
            weights=resampled_weights,
            log_weights=resampled_log_weights,
            ancestor_indices=ancestor_indices,
            **additional_variables
        )