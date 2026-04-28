import tensorflow as tf

from filterflow.base import State
from filters.Diffpf_soft_resampling import SoftResampler


def make_state(batch_size=2, n_particles=5, dim=3, weights=None, dtype=tf.float32):
    particles = tf.reshape(
        tf.cast(tf.range(batch_size * n_particles * dim), dtype),
        [batch_size, n_particles, dim]
    )

    if weights is None:
        if batch_size == 2 and n_particles == 5:
            weights = tf.constant([
                [0.05, 0.15, 0.30, 0.10, 0.40],
                [0.50, 0.10, 0.10, 0.20, 0.10],
            ], dtype=dtype)
        else:
            weights = tf.ones([batch_size, n_particles], dtype=dtype) / tf.cast(n_particles, dtype)

    weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
    log_weights = tf.math.log(weights)

    return State(
        particles=particles,
        weights=weights,
        log_weights=log_weights,
    )


class SoftResamplerTest(tf.test.TestCase):

    def test_output_shapes_and_normalization(self):
        state = make_state()
        flags = tf.constant([True, True])

        resampler = SoftResampler(alpha=0.5, on_log=True)
        out = resampler.apply(state, flags, seed=tf.constant([123, 456], dtype=tf.int32))

        self.assertAllEqual(out.particles.shape, state.particles.shape)
        self.assertAllEqual(out.weights.shape, state.weights.shape)
        self.assertAllEqual(out.log_weights.shape, state.log_weights.shape)
        self.assertAllEqual(out.ancestor_indices.shape, state.ancestor_indices.shape)

        weight_sums = tf.reduce_sum(out.weights, axis=1)
        self.assertAllClose(weight_sums, tf.ones_like(weight_sums), atol=1e-6)

        expected_weights_from_log = tf.exp(out.log_weights)
        expected_weights_from_log /= tf.reduce_sum(expected_weights_from_log, axis=1, keepdims=True)
        self.assertAllClose(out.weights, expected_weights_from_log, atol=1e-6)

    def test_flags_false_keeps_state_unchanged(self):
        state = make_state()
        flags = tf.constant([False, False])

        resampler = SoftResampler(alpha=0.5, on_log=True)
        out = resampler.apply(state, flags, seed=tf.constant([123, 456], dtype=tf.int32))

        self.assertAllClose(out.particles, state.particles, atol=1e-6)
        self.assertAllClose(out.weights, state.weights, atol=1e-6)
        self.assertAllClose(out.log_weights, state.log_weights, atol=1e-6)

        expected_ancestors = tf.tile(
            tf.expand_dims(tf.range(state.n_particles, dtype=state.ancestor_indices.dtype), axis=0),
            [state.batch_size, 1]
        )
        self.assertAllEqual(out.ancestor_indices, expected_ancestors)

    def test_alpha_one_produces_uniform_weights_after_resampling(self):
        state = make_state()
        flags = tf.constant([True, True])

        resampler = SoftResampler(alpha=1.0, on_log=True)
        out = resampler.apply(state, flags, seed=tf.constant([11, 22], dtype=tf.int32))

        expected = tf.ones_like(out.weights) / tf.cast(state.n_particles, out.weights.dtype)
        self.assertAllClose(out.weights, expected, atol=1e-5)

    def test_alpha_zero_importance_correction(self):
        state = make_state()
        flags = tf.constant([True, True])

        resampler = SoftResampler(alpha=0.0, on_log=True)
        out = resampler.apply(state, flags, seed=tf.constant([7, 9], dtype=tf.int32))

        gathered_old_weights = tf.gather(
            state.weights,
            out.ancestor_indices,
            axis=1,
            batch_dims=1
        )
        expected = gathered_old_weights / tf.reduce_sum(gathered_old_weights, axis=1, keepdims=True)

        self.assertAllClose(out.weights, expected, atol=1e-6)

    def test_resampled_particles_follow_ancestor_indices(self):
        state = make_state()
        flags = tf.constant([True, True])

        resampler = SoftResampler(alpha=0.3, on_log=True)
        out = resampler.apply(state, flags, seed=tf.constant([101, 202], dtype=tf.int32))

        expected_particles = tf.gather(
            state.particles,
            out.ancestor_indices,
            axis=1,
            batch_dims=1
        )

        self.assertAllClose(out.particles, expected_particles, atol=1e-6)

    def test_mixed_flags_only_resample_selected_batches(self):
        state = make_state()
        flags = tf.constant([True, False])

        resampler = SoftResampler(alpha=0.5, on_log=True)
        out = resampler.apply(state, flags, seed=tf.constant([42, 24], dtype=tf.int32))

        # batch 1 not resampled
        self.assertAllClose(out.particles[1], state.particles[1], atol=1e-6)
        self.assertAllClose(out.weights[1], state.weights[1], atol=1e-6)
        self.assertAllClose(out.log_weights[1], state.log_weights[1], atol=1e-6)

        expected_identity = tf.range(state.n_particles, dtype=out.ancestor_indices.dtype)
        self.assertAllEqual(out.ancestor_indices[1], expected_identity)

        # batch 0 resampled => weights should still sum to 1
        self.assertAllClose(tf.reduce_sum(out.weights[0]), 1.0, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()