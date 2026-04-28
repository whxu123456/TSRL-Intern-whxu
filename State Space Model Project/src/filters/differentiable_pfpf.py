import tensorflow as tf
import tensorflow_probability as tfp
from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel
from src.filters.EKF_nonlinear_ssm import EKF_NonlinearSSM
from src.filters.nonlinear_pfpf import NonlinearPFPF
from filterflow.base import State
from filterflow.resampling import RegularisedTransform, NeffCriterion

tfd = tfp.distributions
tf.random.set_seed(42)


class DifferentiableNonlinearPFPF(NonlinearPFPF):
    """
    PURE EAGER MODE: No tf.function, 100% robust
    """

    def __init__(
            self,
            config: NonlinearSSMConfig,
            model: NonlinearSSMModel,
            ekf: EKF_NonlinearSSM,
            flow_method='LEDH',
            epsilon=0.5,
            use_optimal_homotopy=False
    ):
        super().__init__(
            config=config,
            model=model,
            ekf=ekf,
            flow_method=flow_method,
            use_optimal_homotopy=use_optimal_homotopy
        )
        self.resampling_criterion = NeffCriterion(threshold=0.5, is_relative=True)
        self.differentiable_resampler = RegularisedTransform(epsilon=epsilon)

    def _differentiable_resample(self, particles, weights):
        """
        Pure eager mode differentiable resampling
        """
        N = tf.shape(particles)[0]
        batch_size = 1

        particles_batched = particles[tf.newaxis, ...]
        weights_batched = weights[tf.newaxis, ...]
        log_weights_batched = tf.math.log(weights_batched + 1e-30)

        state = State(
            particles=particles_batched,
            weights=weights_batched,
            log_weights=log_weights_batched
        )
        flags = tf.constant([True], dtype=tf.bool)
        new_state = self.differentiable_resampler.apply(state=state, flags=flags, seed=42)

        resampled_particles = tf.squeeze(new_state.particles, axis=0)
        uniform_weights = tf.squeeze(new_state.weights, axis=0)
        return resampled_particles, uniform_weights

    def compute_differentiable_log_likelihood(self, y_obs, theta):
        """
        Pure eager mode log-likelihood calculation
        """
        T = y_obs.shape[0]
        log_likelihood = 0.0

        # Reset filter state
        self.particles = self.model.sample_initial_particles(self.cfg.n_particles)
        self.weights = tf.ones([self.cfg.n_particles], dtype=tf.float32) / self.cfg.n_particles
        self.ekf.x_est = tf.identity(self.cfg.x0_mean)
        self.ekf.P = tf.identity(self.cfg.P0)

        # Initial EKF update
        self.ekf.update(y_obs[0])

        # Pure Python loop
        for t in range(1, T):
            t_float = tf.cast(t, tf.float32)
            particles_prev = self.particles
            weights_prev = self.weights

            # Particle flow (eager mode)
            P_pred = self.ekf.predict(t_float)
            eta_0 = self.model.propagate_particles(particles_prev, t_float)
            eta_1, log_det_jacobian = self.run_flow(eta_0, y_obs[t], P_pred)

            # Log-likelihood increment
            log_weights_prev = tf.math.log(weights_prev + 1e-30)
            log_lik_inc = tf.reduce_logsumexp(
                log_weights_prev +
                self.model.measurement_log_prob(y_obs[t], eta_1) +
                self.model.transition_log_prob(eta_1, particles_prev, t_float) -
                self.model.transition_log_prob(eta_0, particles_prev, t_float) +
                log_det_jacobian
            )
            log_likelihood += log_lik_inc

            # Update filter state
            self.weights = self._update_weights(y_obs[t], particles_prev, eta_0, eta_1, log_det_jacobian, t_float)
            self.particles = eta_1

            # Resampling
            ess = self.effective_sample_size()
            if ess < self.cfg.n_particles / 2.0:
                self.particles, self.weights = self._differentiable_resample(self.particles, self.weights)

            self.ekf.update(y_obs[t])

        return log_likelihood