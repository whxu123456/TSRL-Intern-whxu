import numpy as np
import tensorflow as tf
import time


class ParticleFilter:
    """
    Particle Filter.
    """

    def __init__(self, model, num_particles=1000):
        self.model = model
        self.N = num_particles

    def systematic_resampling(self, weights):
        N = self.N
        cumsum = tf.cumsum(weights)
        U1 = tf.random.uniform([], 0, 1 / N)
        U = U1 + tf.range(N, dtype=tf.float32) / N
        indices = tf.searchsorted(cumsum, U)
        return indices

    def run(self, y_obs):
        T = y_obs.shape[0]
        particles = tf.random.normal((self.N,), 0, self.model.sigma ** 2 / np.sqrt(1 - self.model.alpha ** 2))
        weights = tf.ones((self.N,)) / self.N
        x_est = np.zeros(T)
        ess_history = np.zeros(T)

        start_time = time.time()
        for t in range(T):
            noise = tf.random.normal((self.N,), 0, 1)
            particles = self.model.alpha * particles + self.model.sigma * noise

            obs_vars = (self.model.beta ** 2) * tf.exp(particles)
            log_likelihoods = -0.5 * tf.math.log(2 * np.pi * obs_vars) - (y_obs[t] ** 2) / (2 * obs_vars)
            log_weights = tf.math.log(weights + 1e-10) + log_likelihoods

            max_log_w = tf.reduce_max(log_weights)
            weights = tf.exp(log_weights - max_log_w)
            weights /= tf.reduce_sum(weights)

            x_est[t] = tf.reduce_sum(particles * weights)
            ess = 1.0 / tf.reduce_sum(weights ** 2)
            ess_history[t] = ess

            if ess < self.N / 2:
                indices = self.systematic_resampling(weights)
                particles = tf.gather(particles, indices)
                weights = tf.ones((self.N,)) / self.N

        return x_est, ess_history, time.time() - start_time
