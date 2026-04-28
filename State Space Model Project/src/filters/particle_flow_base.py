import tensorflow as tf
import numpy as np
from src.filters.stochastic_particle_flow_base import BaseParticleFlowFilterCore


class BaseParticleFlowFilter(BaseParticleFlowFilterCore):
    """
    Base class for Li (2017) Particle Flow Particle Filter (PF-PF).
    Implements invertible EDH/LEDH particle flows, with optional Dai22 optimal stiffness-mitigating homotopy.
    """

    def __init__(
            self,
            num_particles,
            state_dim,
            meas_dim,
            num_flow_steps=29,
            flow_method='EDH',
            R_cov=None,
            init_step=1e-3,
            step_ratio=1.2,
            use_optimal_homotopy=False
    ):
        """
        Initialize base PF-PF flow class.
        Args:
            num_particles: Number of particles in the filter
            state_dim: Dimension of the state vector
            meas_dim: Dimension of the measurement vector
            num_flow_steps: Number of discrete homotopy steps
            flow_method: Linearization method: 'EDH' (global mean) or 'LEDH' (per-particle)
            R_cov: Measurement noise covariance matrix
            init_step: Initial step size for Li17 exponential lambda schedule
            step_ratio: Step size growth ratio for exponential schedule
            use_optimal_homotopy: Enable Dai22 optimal beta homotopy
        """
        # Initialize stochastic flow core (includes Dai22 optimal beta solver)
        super().__init__(
            model=None,
            num_particles=num_particles,
            num_steps=num_flow_steps,
            resample=False
        )

        # filter dimensions and configuration
        self.N = num_particles
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.num_flow_steps = num_flow_steps
        self.flow_method = flow_method
        self.dt = 1.0 / num_flow_steps
        self.use_optimal_homotopy = use_optimal_homotopy

        # Measurement noise covariance with numerical stability regularization
        self.R_cov = R_cov if R_cov is not None else tf.eye(meas_dim, dtype=tf.float32)
        self.R_inv = tf.linalg.inv(self.R_cov + 1e-12 * tf.eye(meas_dim, dtype=tf.float32))

        # Li17 standard exponential lambda schedule (linear homotopy baseline)
        self.lambda_steps = self._build_lambda_schedule(
            n_lambda=num_flow_steps,
            init_step=init_step,
            ratio=step_ratio
        )

        # Stability diagnostics storage
        self.flow_mags = []
        self.condition_numbers = []

    def compute_covariance(self, x):
        """
        Compute sample covariance matrix of particle cloud.
        Args:
            x: Particle tensor, shape [num_particles, state_dim]
        Returns:
            cov: Sample covariance matrix, shape [state_dim, state_dim]
        """
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        centered = x - mean
        cov = tf.matmul(centered, centered, transpose_a=True) / (tf.cast(tf.shape(x)[0], tf.float32) - 1.0)
        return cov

    def _normalize_log_weights(self, log_w):
        """
        Numerically stable log weight normalization.
        Args:
            log_w: Unnormalized log weights, shape [num_particles]
        Returns:
            normalized_weights: Normalized linear weights, shape [num_particles]
        """
        log_w_max = tf.reduce_max(log_w)
        log_w_shifted = log_w - log_w_max
        w_shifted = tf.exp(log_w_shifted)
        w_sum = tf.reduce_sum(w_shifted)
        return w_shifted / w_sum

    def effective_sample_size(self, weights=None):
        """
        Compute Effective Sample Size (ESS) of particle weights.
        Args:
            weights: Normalized particle weights (uses self.weights if None)
        Returns:
            ess: Scalar effective sample size
        """
        weights = self.weights if weights is None else weights
        return 1.0 / tf.reduce_sum(tf.square(weights))

    def systematic_resample(self, weights):
        """
        Systematic resampling for particle degeneracy mitigation.
        Args:
            weights: Normalized particle weights, shape [num_particles]
        Returns:
            resampled_indices: Indices of resampled particles, shape [num_particles]
        """
        n = self.num_particles
        positions = (tf.range(n, dtype=tf.float32) + tf.random.uniform([], 0.0, 1.0)) / tf.cast(n, tf.float32)
        cdf = tf.cumsum(weights)
        idx = tf.searchsorted(cdf, positions, side='left')
        return tf.clip_by_value(idx, 0, n - 1)

    def _build_lambda_schedule(self, n_lambda=29, init_step=1e-3, ratio=1.2):
        """
        Build exponentially spaced lambda schedule for linear homotopy.
        Args:
            n_lambda: Number of flow steps
            init_step: Initial step size
            ratio: Step size growth ratio
        Returns:
            lambdas: Lambda sequence from 0 to 1, shape [n_lambda+1]
        """
        steps = [init_step]
        for _ in range(1, n_lambda):
            steps.append(steps[-1] * ratio)
        steps = np.asarray(steps, dtype=np.float32)
        steps = steps / np.sum(steps)  # Normalize to sum to 1
        lambdas = np.concatenate([[0.0], np.cumsum(steps)], axis=0)
        lambdas[-1] = 1.0  # Enforce final lambda = 1 exactly
        return tf.constant(lambdas, dtype=tf.float32)

    def _record_diagnostics(self, drift, jacobian_matrix=None):
        """
        Record flow magnitude and Jacobian condition number for stability analysis.
        Args:
            drift: Flow drift term, shape [num_particles, state_dim]
            jacobian_matrix: Flow step Jacobian, shape [state_dim, state_dim] (EDH) or [num_particles, state_dim, state_dim] (LEDH)
        """
        # Record mean absolute flow magnitude
        mean_flow_mag = tf.reduce_mean(tf.abs(drift))
        self.flow_mags.append(mean_flow_mag)

        # Record mean Jacobian condition number
        if jacobian_matrix is not None:
            s = tf.linalg.svd(jacobian_matrix, compute_uv=False)
            if len(s.shape) > 2:  # Batch of matrices (LEDH)
                max_s = tf.reduce_max(s, axis=1)
                min_s = tf.reduce_min(s, axis=1)
            else:  # Single matrix (EDH)
                max_s = tf.reduce_max(s)
                min_s = tf.reduce_min(s)
            mean_cond = tf.reduce_mean(max_s / (min_s + 1e-12))
            self.condition_numbers.append(mean_cond)
        else:
            self.condition_numbers.append(1.0)

    def compute_flow_params(self, eta_linearize, z, P, lam):
        """
        Compute flow matrix A(lambda) and bias b(lambda) for invertible particle flow.
        Supports both EDH (global linearization) and LEDH (per-particle linearization).
        Args:
            eta_linearize: Linearization points, shape [1, state_dim] (EDH) or [num_particles, state_dim] (LEDH)
            z: Observation vector, shape [meas_dim]
            P: Predictive covariance matrix, shape [state_dim, state_dim]
            lam: Current homotopy parameter (lambda or Dai22 beta), scalar
        Returns:
            A: Flow matrix, shape [1, state_dim, state_dim] (EDH) or [num_particles, state_dim, state_dim] (LEDH)
            b: Flow bias term, shape [1, state_dim] (EDH) or [num_particles, state_dim] (LEDH)
        """
        batch_size = tf.shape(eta_linearize)[0]

        # Get measurement Jacobian and mean from model
        H = self.model.measurement_jacobian(eta_linearize)  # [B, S, D]
        h_val = self.model.measurement_model(eta_linearize)  # [B, S]

        # Broadcast covariance matrices to batch dimension
        P_batch = tf.broadcast_to(P[None, :, :], [batch_size, self.state_dim, self.state_dim])
        R_batch = tf.broadcast_to(self.R_cov[None, :, :], [batch_size, self.meas_dim, self.meas_dim])

        # Compute flow matrix A(lambda) Eq.10
        HPHt = tf.matmul(tf.matmul(H, P_batch), H, transpose_b=True)
        M = lam * HPHt + R_batch
        M_inv = tf.linalg.inv(M + 1e-12 * tf.eye(self.meas_dim, dtype=tf.float32)[None, :, :])
        PHt = tf.matmul(P_batch, H, transpose_b=True)
        A = -0.5 * tf.matmul(PHt, tf.matmul(M_inv, H))

        # Compute flow bias b(lambda) Eq.11
        H_eta = tf.einsum('bij,bj->bi', H, eta_linearize)
        e = h_val - H_eta
        innov = z[None, :] - e

        PHt_Rinv = tf.matmul(PHt, self.R_inv[None, :, :])
        term1 = tf.einsum('bij,bj->bi', PHt_Rinv, innov)
        A_term1 = tf.einsum('bij,bj->bi', A, term1)
        part1 = term1 + lam * A_term1

        part2 = tf.einsum('bij,bj->bi', A, eta_linearize)
        bracket = part1 + part2

        A_bracket = tf.einsum('bij,bj->bi', A, bracket)
        b = bracket + 2.0 * lam * A_bracket

        return A, b

    def _flow_step_edh(self, x, A, b, dl):
        """Single EDH affine flow step (global linearization per Li17)."""
        drift = tf.matmul(x, A[0], transpose_b=True) + b
        return x + dl * drift

    def _flow_step_ledh(self, x, A, b, dl):
        """Single LEDH affine flow step (per-particle linearization per Li17)."""
        drift = tf.einsum('bij,bj->bi', A, x) + b
        return x + dl * drift

    def run_flow(self, eta, z_obs, P):
        """
        Run full invertible particle flow, with support for Li17 linear homotopy or Dai22 optimal homotopy.
        Preserves Li17's invertible mapping property for exact weight calculation.
        Args:
            eta: Initial prior particles, shape [num_particles, state_dim]
            z_obs: Observation vector, shape [meas_dim]
            P: Predictive covariance matrix, shape [state_dim, state_dim]
        Returns:
            eta_1: Final proposal particles after flow, shape [num_particles, state_dim]
            log_det_sum: Log absolute Jacobian determinant of the flow, shape [num_particles]
        """
        eta_curr = eta
        log_det_sum = tf.zeros([self.num_particles], dtype=tf.float32)

        # Set linearization points per flow mode
        if self.flow_method == 'LEDH':
            eta_aux = eta  # Per-particle linearization (LEDH)
        else:  # EDH
            eta_aux = tf.reduce_mean(eta, axis=0, keepdims=True)  # Global mean linearization (EDH)

        # Select homotopy schedule: Dai22 optimal beta or Li17 linear lambda
        if self.use_optimal_homotopy:
            # Solve Dai22 optimal beta homotopy (stiffness mitigation)
            prior_mean = tf.reduce_mean(eta, axis=0)
            beta_vals, _ = self.solve_optimal_beta(z_obs, prior_mean, P)
            flow_steps = beta_vals
        else:
            # Use Li17 standard exponential linear lambda schedule
            flow_steps = self.lambda_steps

        # Iterate over homotopy steps
        for k in range(len(flow_steps) - 1):
            lam_prev = flow_steps[k]
            lam_curr = flow_steps[k + 1]
            dl = lam_curr - lam_prev  # Step size in homotopy parameter

            # Compute flow parameters at current homotopy value
            A, b = self.compute_flow_params(eta_aux, z_obs, P, lam_curr)

            # Update linearization points and particles
            if self.flow_method == 'EDH':
                # EDH: global linearization, same Jacobian for all particles
                eta_aux = self._flow_step_edh(eta_aux, A, b, dl)
                eta_curr = self._flow_step_edh(eta_curr, A, b, dl)

                # Log determinant (identical for all particles in EDH)
                J = tf.eye(self.state_dim, dtype=tf.float32) + dl * A[0]
                log_det = tf.math.log(tf.abs(tf.linalg.det(J)) + 1e-30)
                log_det_sum += log_det

                # Record stability diagnostics
                drift = tf.matmul(eta_curr, A[0], transpose_b=True) + b
                self._record_diagnostics(drift, J[None, :, :])

            else:  # LEDH
                # LEDH: per-particle linearization, individual Jacobian per particle
                eta_aux = self._flow_step_ledh(eta_aux, A, b, dl)
                eta_curr = self._flow_step_ledh(eta_curr, A, b, dl)

                # Log determinant (per-particle for LEDH)
                I = tf.eye(self.state_dim, dtype=tf.float32)[None, :, :]
                J = I + dl * A
                log_det_sum += tf.math.log(tf.abs(tf.linalg.det(J)) + 1e-30)

                # Record stability diagnostics
                drift = tf.einsum('bij,bj->bi', A, eta_curr) + b
                self._record_diagnostics(drift, J)

        return eta_curr, log_det_sum

    # Abstract Methods for Subclass Implementation
    def predict(self):
        """Prior prediction step (implemented in subclass)."""
        raise NotImplementedError("Subclasses must implement predict() method")

    def update(self, y_obs):
        """Measurement update step (implemented in subclass)."""
        raise NotImplementedError("Subclasses must implement update() method")