import tensorflow as tf
EPS = 1e-12
def ess_from_weights(weights):
    """
    weights: [T, B, N]
    return: [T, B]
    """
    return 1.0 / (tf.reduce_sum(tf.square(weights), axis=-1) + EPS)


def ess_ratio_from_weights(weights):
    """
    weights: [T, B, N]
    return: [T, B]
    """
    n_particles = tf.cast(tf.shape(weights)[-1], tf.float32)
    return ess_from_weights(weights) / n_particles


def weighted_particle_mean(weights, particles):
    """
    weights: [T, B, N]
    particles: [T, B, N, dx]
    return: [T, B, dx]
    """
    return tf.reduce_sum(weights[..., None] * particles, axis=2)


def weighted_cov_trace(weights, particles):
    """
    weights: [T, B, N]
    particles: [T, B, N, dx]
    return: [T, B]
    """
    mean = weighted_particle_mean(weights, particles)  # [T, B, dx]
    centered = particles - mean[:, :, None, :]
    sq_norm = tf.reduce_sum(tf.square(centered), axis=-1)  # [T, B, N]
    return tf.reduce_sum(weights * sq_norm, axis=-1)


def avg_pairwise_sq_distance(particles):
    """
    particles: [T, B, N, dx]
    return: [T, B]
    """
    xi = particles[:, :, :, None, :]   # [T, B, N, 1, dx]
    xj = particles[:, :, None, :, :]   # [T, B, 1, N, dx]
    d2 = tf.reduce_sum(tf.square(xi - xj), axis=-1)  # [T, B, N, N]
    return tf.reduce_mean(d2, axis=[-1, -2])


def particle_diversity(weights, particles, mode="cov_trace"):
    if mode == "cov_trace":
        return weighted_cov_trace(weights, particles)
    elif mode == "pairwise":
        return avg_pairwise_sq_distance(particles)
    else:
        raise ValueError(f"Unknown diversity mode: {mode}")


def infer_resampled_steps(log_weights, atol=1e-6):
    n_particles = tf.cast(tf.shape(log_weights)[-1], tf.float32)
    uniform_logw = -tf.math.log(n_particles)
    max_dev = tf.reduce_max(tf.abs(log_weights - uniform_logw), axis=-1)  # [T, B]
    return max_dev < atol


def infer_soft_resampling_frequency(weights, threshold_ratio):
    ess_ratio = ess_ratio_from_weights(weights)  # [T, B]
    B = tf.shape(ess_ratio)[1]

    prev_flag = ess_ratio[:-1] <= threshold_ratio  # [T-1, B]
    first = tf.zeros([1, B], dtype=tf.bool)
    flags = tf.concat([first, prev_flag], axis=0)
    return flags


def summarize_states(
    states_series,
    method,
    resampling_neff=0.5,
    diversity_mode="cov_trace",
):
    """
    states_series fields:
      particles              [T, B, N, dx]
      weights                [T, B, N]
      log_weights            [T, B, N]
      log_likelihoods        [T, B]
      optional:
      resampling_flags       [T, B]
      ess_before_resampling  [T, B]
    """
    particles = tf.convert_to_tensor(states_series.particles)
    weights = tf.convert_to_tensor(states_series.weights)
    log_weights = tf.convert_to_tensor(states_series.log_weights)
    log_likelihoods = tf.convert_to_tensor(states_series.log_likelihoods)

    ess_ratio = ess_ratio_from_weights(weights)  # [T, B]
    diversity = particle_diversity(weights, particles, mode=diversity_mode)  # [T, B]

    if hasattr(states_series, "resampling_flags") and states_series.resampling_flags is not None:
        resampled = tf.convert_to_tensor(states_series.resampling_flags, dtype=tf.bool)
    else:
        method = method.upper()
        if method in ("NO",):
            resampled = tf.zeros_like(ess_ratio, dtype=tf.bool)
        elif method in ("OT",):
            resampled = infer_resampled_steps(log_weights)
        elif method in ("SOFT",):
            resampled = infer_soft_resampling_frequency(weights, resampling_neff)
        else:
            raise ValueError(f"Unknown method: {method}")

    final_loglik = log_likelihoods[-1]  # [B]

    out = {
        "loglik_per_batch": final_loglik.numpy(),
        "mean_loglik": float(tf.reduce_mean(final_loglik).numpy()),
        "std_loglik": float(tf.math.reduce_std(final_loglik).numpy()),
        "mean_ess_ratio": float(tf.reduce_mean(ess_ratio).numpy()),
        "min_ess_ratio": float(tf.reduce_min(ess_ratio).numpy()),
        "resampling_frequency": float(tf.reduce_mean(tf.cast(resampled, tf.float32)).numpy()),
        "mean_diversity": float(tf.reduce_mean(diversity).numpy()),
        "final_diversity": float(tf.reduce_mean(diversity[-1]).numpy()),
    }

    if hasattr(states_series, "ess_before_resampling") and states_series.ess_before_resampling is not None:
        ess_before = tf.convert_to_tensor(states_series.ess_before_resampling)
        out["mean_ess_before_resampling"] = float(tf.reduce_mean(ess_before).numpy())
        out["min_ess_before_resampling"] = float(tf.reduce_min(ess_before).numpy())

    return out