import tensorflow as tf
from src.filters.particle_flow_base import BaseParticleFlowFilter


class AcousticPFPF(BaseParticleFlowFilter):
    """
    Particle Flow Particle Filter for Li (2017) acoustic tracking scenario.
    Supports both standard linear homotopy (Li17) and Dai22 optimal stiffness-mitigating homotopy.
    Compatible with both EDH (global linearization) and LEDH (per-particle linearization) flow modes.
    """

    def __init__(self, config, model, mode='LEDH', use_optimal_homotopy=False):
        """
        config: AcousticConfig object with scenario parameters
        model: AcousticModel object with dynamics and measurement models
        mode: Flow linearization mode, 'EDH' or 'LEDH'
        use_optimal_homotopy: If True, use Dai22 optimal beta homotopy; if False, use Li17 linear lambda schedule
        """
        state_dim = config.state_dim
        meas_dim = config.meas_dim
        num_particles = config.n_particles
        R_cov = config.R_cov

        super().__init__(
            num_particles=num_particles,
            state_dim=state_dim,
            meas_dim=meas_dim,
            num_flow_steps=29,
            flow_method=mode,
            R_cov=R_cov,
            use_optimal_homotopy=use_optimal_homotopy
        )

        # Acoustic scenario specific parameters
        self.cfg = config
        self.model = model
        self.mode = mode
        self.use_optimal_homotopy = use_optimal_homotopy

        # Link model to base flow class for Jacobian/measurement calculations
        self.model = model

        # Initialize particles and uniform weights
        self.particles = self.model.sample_initial_particles(self.cfg.n_particles)
        self.weights = tf.ones([self.cfg.n_particles], dtype=tf.float32) / self.cfg.n_particles

    def run_step(self, z_obs, P_pred):
        """
        Execute one full PF-PF prediction and update step.
        Args:
            z_obs: [meas_dim] observation vector at current time step
            P_pred: [state_dim, state_dim] EKF predictive covariance matrix
        Returns:
            x_est: [state_dim] weighted mean state estimate from posterior particles
        """
        particles_prev = self.particles

        # 1. Prior propagation: sample from transition density
        eta_0 = self.model.propagate_particles(particles_prev)

        # 2. Invertible particle flow proposal
        eta_1, log_det = self.run_flow(eta_0, z_obs, P_pred)
        eta_1 = self.model.reflect_boundaries(eta_1)

        # 3. Importance weight update per invertible mapping formula
        self.weights = self._update_weights(
            z_obs=z_obs,
            particles_prev=particles_prev,
            eta_0=eta_0,
            eta_1=eta_1,
            log_det_jacobian=log_det
        )

        # Update particle set to flow output
        self.particles = eta_1

        # 4. Resample when effective sample size drops below threshold
        ess = self.effective_sample_size()
        if ess < self.cfg.n_particles / 2.0:
            indices = self.systematic_resample(self.weights)
            self.particles = tf.gather(self.particles, indices)
            self.weights = tf.ones([self.cfg.n_particles], dtype=tf.float32) / self.cfg.n_particles

        # 5. Compute weighted posterior mean estimate
        return tf.reduce_sum(self.particles * self.weights[:, None], axis=0)

    def _update_weights(self, z_obs, particles_prev, eta_0, eta_1, log_det_jacobian):
        """
        Importance weight update per invertible particle flow formula.
        Preserves exact weight calculation for both linear and optimal homotopy schedules.
        Args:
            z_obs: Current observation vector
            particles_prev: Particles from previous time step posterior
            eta_0: Prior particles after propagation (flow input)
            eta_1: Proposal particles after flow (flow output)
            log_det_jacobian: Log absolute determinant of flow Jacobian
        Returns:
            normalized_weights: Normalized importance weights for particles
        """
        # Stabilize log weights with small epsilon to avoid log(0)
        log_w_prev = tf.math.log(self.weights + 1e-30)

        # Compute log probability terms for weight update
        log_likelihood = self.model.measurement_log_prob(z_obs, eta_1)
        log_trans_eta1 = self.model.transition_log_prob(eta_1, particles_prev)
        log_trans_eta0 = self.model.transition_log_prob(eta_0, particles_prev)

        # Exact weight update formula for invertible flow
        log_w = log_w_prev + log_likelihood + log_trans_eta1 - log_trans_eta0 + log_det_jacobian

        return self._normalize_log_weights(log_w)