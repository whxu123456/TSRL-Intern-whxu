import tensorflow as tf
import numpy as np
from src.filters.stochastic_particle_flow_base import BaseParticleFlowFilterCore


class ParticleFlowFilterDai22(BaseParticleFlowFilterCore):
    """
    Dai22-specific stochastic Particle Flow Filter.
    Inherits all utilities from BaseParticleFlowFilterCore.
    """

    def __init__(
            self,
            model,
            num_particles=50,
            num_steps=100,
            resample=False,
            diffusion_scale=0.03,
            jitter=1e-6,
            drift_clip=10.0
    ):
        super().__init__(
            model=model,
            num_particles=num_particles,
            num_steps=num_steps,
            resample=resample,
            diffusion_scale=diffusion_scale,
            jitter=jitter,
            drift_clip=drift_clip
        )

    def run_flow(self, z, beta_vals, beta_dot_vals, prior_particles, prior_mean, prior_cov):
        """stochastic particle flow (Theorem 2.1)."""
        particles = tf.identity(tf.convert_to_tensor(prior_particles, dtype=tf.float32))
        H_log_p0, H_log_h = self._compute_hessians(z, prior_mean, prior_cov)

        Q = self._get_diffusion_cov(dtype=tf.float32)
        q_chol = tf.linalg.cholesky(Q)
        P_inv = self._safe_inverse(prior_cov)
        R = self.model.get_measurement_cov(prior_mean)
        R_inv = self._safe_inverse(R)
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        prior_mean = tf.convert_to_tensor(prior_mean, dtype=tf.float32)

        for k in range(self.num_steps):
            beta = tf.cast(beta_vals[k], tf.float32)
            beta_dot = tf.cast(beta_dot_vals[k], tf.float32)
            alpha = 1.0 - beta
            alpha_dot = -beta_dot

            # Log-likelihood gradient
            with tf.GradientTape() as tape:
                tape.watch(particles)
                hx = self._measurement_eval(particles)
                innov = z[None, :] - hx
                log_h = -0.5 * tf.einsum('bi,ij,bj->b', innov, R_inv, innov)
            grad_log_h = tape.gradient(log_h, particles)

            # Log-prior gradient (Gaussian approx)
            grad_log_p0 = -tf.matmul(particles - prior_mean[None, :], P_inv)
            grad_log_p = grad_log_p0 + beta * grad_log_h

            # Compute S matrix and K1/K2
            S = H_log_p0 + beta * H_log_h
            S = self._symmetrize(S)
            S_inv = self._safe_inverse(S)

            denom = alpha + beta + 1e-12
            num = alpha * beta_dot - alpha_dot * beta

            K1 = 0.5 * Q + \
                 (num / (2.0 * denom)) * tf.matmul(tf.matmul(S_inv, H_log_h), S_inv) - \
                 ((alpha_dot + beta_dot) / (2.0 * denom)) * S_inv
            K2 = -(num / denom) * S_inv

            # Drift + diffusion update
            drift = tf.matmul(grad_log_p, K1, transpose_b=True) + tf.matmul(grad_log_h, K2, transpose_b=True)
            drift = tf.clip_by_value(drift, -self.drift_clip, self.drift_clip)

            dw = tf.random.normal(shape=tf.shape(particles), dtype=tf.float32) * tf.sqrt(self.dlambda)
            diffusion = tf.matmul(dw, q_chol, transpose_b=True)

            particles = particles + drift * self.dlambda + diffusion
            particles = self._apply_boundaries(particles)

        return particles

    def compute_stiffness_ratio_path(self, z, beta_vals, prior_mean, prior_cov):
        """Compute stiffness ratio R_stiff along homotopy path."""
        beta_vals = tf.convert_to_tensor(beta_vals, dtype=tf.float32)
        H_log_p0, H_log_h = self._compute_hessians(z, prior_mean, prior_cov)
        H_log_p0_np = H_log_p0.numpy()
        H_log_h_np = H_log_h.numpy()
        beta_np = beta_vals.numpy()

        R_stiff = np.zeros_like(beta_np, dtype=np.float32)
        for k in range(len(beta_np)):
            b = beta_np[k]
            M = -H_log_p0_np - b * H_log_h_np
            M = 0.5 * (M + M.T)
            try:
                eigvals = np.linalg.eigvalsh(M)
                abs_eigs = np.abs(eigvals)
                max_eig = np.max(abs_eigs)
                min_eig = np.min(abs_eigs)
                R_stiff[k] = max_eig / min_eig if min_eig > 1e-12 else np.inf
            except np.linalg.LinAlgError:
                R_stiff[k] = np.inf
        return tf.convert_to_tensor(R_stiff, dtype=tf.float32)

    def update(self, predicted_particles, z, use_optimal_homotopy=True):
        """Single update step."""
        prior_mean, prior_cov = self._sample_covariance(predicted_particles)
        if use_optimal_homotopy:
            beta_vals, beta_dot_vals = self.solve_optimal_beta(z, prior_mean, prior_cov)
        else:
            beta_vals, beta_dot_vals = self._linear_beta_schedule()

        updated_particles = self.run_flow(
            z=z, beta_vals=beta_vals, beta_dot_vals=beta_dot_vals,
            prior_particles=predicted_particles, prior_mean=prior_mean, prior_cov=prior_cov
        )

        if self.resample:
            updated_particles = self.systematic_resample(updated_particles)
        post_mean, post_cov = self._sample_covariance(updated_particles)
        return updated_particles, post_mean, post_cov, beta_vals, beta_dot_vals

    def filter_step(self, particles, z):
        """One sequential filtering step."""
        predicted_particles = self.predict(particles)
        updated_particles, post_mean, post_cov, beta_vals, beta_dot_vals = self.update(predicted_particles, z)
        return updated_particles, {
            "predicted_particles": predicted_particles,
            "prior_mean": post_mean,
            "prior_cov": post_cov,
            "beta_vals": beta_vals,
            "beta_dot_vals": beta_dot_vals,
            "estimate": post_mean,
            "covariance": post_cov
        }

    def filter_sequence(self, initial_particles, measurements):
        """Run sequential filtering over a measurement sequence."""
        particles = tf.identity(initial_particles)
        particles_seq = []
        info_seq = []
        estimates = []
        covariances = []

        for z in measurements:
            z = tf.convert_to_tensor(z, dtype=tf.float32)
            particles, info = self.filter_step(particles, z)
            particles_seq.append(particles)
            info_seq.append(info)
            estimates.append(info["estimate"])
            covariances.append(info["covariance"])

        estimates = tf.stack(estimates, axis=0)
        covariances = tf.stack(covariances, axis=0)
        return particles_seq, info_seq, estimates, covariances