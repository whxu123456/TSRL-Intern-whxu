import tensorflow as tf
import numpy as np
from src.filters.particle_flow_base import BaseParticleFlowFilter
from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel
from src.filters.EKF_nonlinear_ssm import EKF_NonlinearSSM


class NonlinearPFPF(BaseParticleFlowFilter):
    """
    Li (2017) Reversible Particle Flow Particle Filter (PF-PF)
    Implementation for Section 3.1 nonlinear state space model, supports EDH/LEDH flow modes
    """
    def __init__(
            self,
            config: NonlinearSSMConfig,
            model: NonlinearSSMModel,
            ekf: EKF_NonlinearSSM,
            flow_method='LEDH',
            use_optimal_homotopy=False
    ):
        """
        Initialize PF-PF filter
        Args:
            config: Nonlinear model configuration
            model: Nonlinear model instance
            ekf: EKF instance for predicting covariance matrix P
            flow_method: Particle flow linearization method, 'EDH' (global mean) or 'LEDH' (per-particle)
            use_optimal_homotopy: Whether to enable Dai22 optimal homotopy (optional extension)
        """
        # Initialize base particle flow class
        super().__init__(
            num_particles=config.n_particles,
            state_dim=config.state_dim,
            meas_dim=config.meas_dim,
            num_flow_steps=29,
            flow_method=flow_method,
            R_cov=config.R_cov,
            use_optimal_homotopy=use_optimal_homotopy
        )

        # Model binding
        self.cfg = config
        self.model = model
        self.ekf = ekf
        self.flow_method = flow_method

        # Initialize particles and weights
        self.particles = self.model.sample_initial_particles(self.cfg.n_particles)
        self.weights = tf.ones([self.cfg.n_particles], dtype=tf.float32) / self.cfg.n_particles

    def run_step(self, z_obs, t):
        """
        Perform single-step PF-PF filtering (prediction + update)
        Args:
            z_obs: Current time step observation [meas_dim]
            t: Current time step
        Returns:
            x_est: Weighted posterior state estimate [state_dim]
        """
        particles_prev = self.particles

        # 1. EKF prediction step, get predicted covariance matrix P for particle flow
        P_pred = self.ekf.predict(t)

        # 2. Prior propagation: sample prior particles eta_0 from transition density
        eta_0 = self.model.propagate_particles(particles_prev, t)

        # 3. Reversible particle flow: map prior particles to posterior region, get proposal particles eta_1
        eta_1, log_det_jacobian = self.run_flow(eta_0, z_obs, P_pred)

        # 4. Reversible mapping weight update (Li2017 core formula)
        self.weights = self._update_weights(
            z_obs=z_obs,
            particles_prev=particles_prev,
            eta_0=eta_0,
            eta_1=eta_1,
            log_det_jacobian=log_det_jacobian,
            t=t
        )

        # Update particle set to flow output
        self.particles = eta_1

        # 5. Resampling: execute when effective sample size is below threshold
        ess = self.effective_sample_size()
        if ess < self.cfg.n_particles / 2.0:
            indices = self.systematic_resample(self.weights)
            self.particles = tf.gather(self.particles, indices)
            self.weights = tf.ones([self.cfg.n_particles], dtype=tf.float32) / self.cfg.n_particles

        # 6. EKF update step, prepare for next time step prediction
        self.ekf.update(z_obs)

        # 7. Compute weighted posterior mean estimate
        x_est = tf.reduce_sum(self.particles * self.weights[:, None], axis=0)
        return x_est

    def _update_weights(self, z_obs, particles_prev, eta_0, eta_1, log_det_jacobian, t):
        """
        Li (2017) Reversible particle flow weight update formula
        Formula: w_k ∝ w_{k-1} * p(z|eta_1) * p(eta_1|x_prev) / p(eta_0|x_prev) * |det(J)|
        Args:
            z_obs: Current observation
            particles_prev: Previous time step posterior particles
            eta_0: Particles after prior propagation (flow input)
            eta_1: Proposal particles from particle flow output
            log_det_jacobian: Log absolute value of flow mapping Jacobian determinant
            t: Current time step
        Returns:
            normalized_weights: Normalized importance weights
        """
        # Numerical stability: avoid log(0)
        log_w_prev = tf.math.log(self.weights + 1e-30)

        # Compute log probabilities of weight terms
        log_likelihood = self.model.measurement_log_prob(z_obs, eta_1)
        log_trans_eta1 = self.model.transition_log_prob(eta_1, particles_prev, t)
        log_trans_eta0 = self.model.transition_log_prob(eta_0, particles_prev, t)

        # Reversible flow weight core formula
        log_w = log_w_prev + log_likelihood + log_trans_eta1 - log_trans_eta0 + log_det_jacobian

        # Numerically stable normalization
        return self._normalize_log_weights(log_w)

    def run_filter(self, y_obs):
        """
        Run complete PF-PF filtering
        Args:
            y_obs: Observation sequence [T, meas_dim]
        Returns:
            x_est_seq: State estimation sequence [T, state_dim]
            ess_seq: Per-step effective sample size sequence [T]
            run_time: Total runtime
        """
        import time
        T = y_obs.shape[0]
        x_est_seq = np.zeros((T, self.cfg.state_dim), dtype=np.float32)
        ess_seq = np.zeros(T, dtype=np.float32)
        start_time = time.time()

        # Initial time step
        self.ekf.update(y_obs[0])
        x_est_seq[0] = tf.reduce_sum(self.particles * self.weights[:, None], axis=0).numpy()
        ess_seq[0] = self.effective_sample_size().numpy()

        # Recursive filtering
        for t in range(1, T):
            x_est = self.run_step(y_obs[t], t)
            x_est_seq[t] = x_est.numpy()
            ess_seq[t] = self.effective_sample_size().numpy()

        run_time = time.time() - start_time
        return x_est_seq, ess_seq, run_time


    def run_filter_with_loglik(self, y_obs):
        """
        Run PF-PF and return filtering estimates plus log likelihood estimate.
        This is for PMMH.
        """
        import time

        y_obs = tf.convert_to_tensor(y_obs, dtype=tf.float32)
        T = int(y_obs.shape[0])
        N = self.cfg.n_particles

        x_est_seq = np.zeros((T, self.cfg.state_dim), dtype=np.float32)
        ess_seq = np.zeros(T, dtype=np.float32)

        start_time = time.time()

        # reset filter state
        self.particles = self.model.sample_initial_particles(N)
        self.weights = tf.ones([N], dtype=tf.float32) / tf.cast(N, tf.float32)

        self.ekf.x_est = tf.identity(self.cfg.x0_mean)
        self.ekf.P = tf.identity(self.cfg.P0)
        self.ekf.x_pred = tf.identity(self.ekf.x_est)
        self.ekf.P_pred = tf.identity(self.ekf.P)

        loglik = tf.constant(0.0, dtype=tf.float32)

        # t = 0 likelihood
        log_w0 = (
            tf.math.log(self.weights + 1e-30)
            + self.model.measurement_log_prob(y_obs[0], self.particles)
        )
        loglik += tf.reduce_logsumexp(log_w0)

        self.weights = self._normalize_log_weights(log_w0)

        ess = self.effective_sample_size()
        if ess < N / 2.0:
            indices = self.systematic_resample(self.weights)
            self.particles = tf.gather(self.particles, indices)
            self.weights = tf.ones([N], dtype=tf.float32) / tf.cast(N, tf.float32)

        self.ekf.update(y_obs[0])

        x_est_seq[0] = tf.reduce_sum(self.particles * self.weights[:, None], axis=0).numpy()
        ess_seq[0] = self.effective_sample_size().numpy()

        # t >= 1
        for t in range(1, T):
            particles_prev = self.particles
            log_w_prev = tf.math.log(self.weights + 1e-30)

            P_pred = self.ekf.predict(t)

            eta_0 = self.model.propagate_particles(particles_prev, t)
            eta_1, log_det_jacobian = self.run_flow(eta_0, y_obs[t], P_pred)

            log_w_unnorm = (
                log_w_prev
                + self.model.measurement_log_prob(y_obs[t], eta_1)
                + self.model.transition_log_prob(eta_1, particles_prev, t)
                - self.model.transition_log_prob(eta_0, particles_prev, t)
                + log_det_jacobian
            )

            # incremental log likelihood
            loglik += tf.reduce_logsumexp(log_w_unnorm)

            self.weights = self._normalize_log_weights(log_w_unnorm)
            self.particles = eta_1

            ess = self.effective_sample_size()
            if ess < N / 2.0:
                indices = self.systematic_resample(self.weights)
                self.particles = tf.gather(self.particles, indices)
                self.weights = tf.ones([N], dtype=tf.float32) / tf.cast(N, tf.float32)

            self.ekf.update(y_obs[t])

            x_est_seq[t] = tf.reduce_sum(self.particles * self.weights[:, None], axis=0).numpy()
            ess_seq[t] = self.effective_sample_size().numpy()

        run_time = time.time() - start_time

        return x_est_seq, ess_seq, loglik.numpy(), run_time