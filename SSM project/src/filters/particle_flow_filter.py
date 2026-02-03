import tensorflow as tf
import numpy as np


class SVParticleFlowFilter:
    """
    Particle Flow Filter for Stochastic Volatility Model.
    Test EDH, LEDH, and Kernel methods.
    """

    def __init__(self, model, num_particles=100, num_steps=50, method='EDH'):
        self.model = model
        self.N = num_particles
        self.steps = num_steps
        self.method = method
        self.dt = 1/num_steps
        # log(w^2) mean approx -1.27, var approx np.pi**2/2
        self.obs_noise_mean = -1.27
        self.obs_noise_var = np.pi**2/2

        # Initialize particles
        init_std = model.sigma/np.sqrt(1 - model.alpha**2)
        self.particles = tf.random.normal((self.N, 1), 0, init_std, dtype=tf.float32)

        self.flow_mags = []
        self.condition_numbers = []

    def predict(self):
        noise = tf.random.normal(self.particles.shape, 0, self.model.sigma)
        self.particles = self.model.alpha * self.particles + noise
        # Reset diagnostics for the new step
        self.flow_mags = []
        self.condition_numbers = []

    def compute_covariance(self, x):
        mean = tf.reduce_mean(x, axis=0)
        centered = x - mean
        cov = tf.matmul(centered, centered, transpose_a=True)/(self.N-1)
        return cov

    def _record_diagnostics(self, drift, jacobian_matrix=None):
        # Record Flow Magnitude
        mag = tf.reduce_mean(tf.abs(drift))
        self.flow_mags.append(mag)

        # Record Jacobian Condition Number
        if jacobian_matrix is not None:
            # SV model is 1D, so condition number is trivially 1.0 if scalar != 0
            # For generality (if extended to >1D), use SVD
            s = tf.linalg.svd(jacobian_matrix, compute_uv=False)
            # Handle batch of matrices for LEDH
            if len(s.shape) > 1:
                max_s = tf.reduce_max(s, axis=1)
                min_s = tf.reduce_min(s, axis=1)
            else:
                max_s = tf.reduce_max(s)
                min_s = tf.reduce_min(s)

            cond = tf.reduce_mean(max_s/(min_s + 1e-12))
            self.condition_numbers.append(cond)
        else:
            self.condition_numbers.append(1)


    def flow_edh_exact(self, y_obs):
        """
        Pure EDH without Log-transform. (Daum(10))
        Linearizes h(x) = beta * exp(x/2) around the mean of particles.
        """

        R = tf.constant([[1]], dtype=tf.float32)
        lam = 0
        for s in range(self.steps):
            P = self.compute_covariance(self.particles)
            x_bar = tf.reduce_mean(self.particles, axis=0, keepdims=True)
            H = 0.5 * self.model.beta * tf.exp(x_bar/2)
            h_bar = self.model.beta * tf.exp(x_bar/2)
            HPH = tf.matmul(H, tf.matmul(P, H, transpose_b=True))
            inv_term = tf.linalg.inv(lam * HPH + R)
            PHt = tf.matmul(P, H, transpose_b=True)
            A = -0.5 * tf.matmul(PHt, tf.matmul(inv_term, H))
            e = h_bar - tf.matmul(x_bar, H, transpose_b=True)
            innov = y_obs - e
            R_inv = tf.linalg.inv(R)
            PHt_Rinv = tf.matmul(PHt, R_inv)
            term_innov = tf.matmul(PHt_Rinv, innov)
            I = tf.eye(1)
            term1 = tf.matmul(I + lam * A, term_innov)
            term2 = tf.matmul(A, x_bar, transpose_b=True)
            b = tf.matmul(I + 2 * lam * A, term1 + term2)
            drift = tf.matmul(self.particles, A, transpose_b=True) + tf.transpose(b)
            self.particles += self.dt * drift
            lam += self.dt
            # Diagnostics
            J = I + self.dt * A
            self._record_diagnostics(drift, J)

    def flow_ledh_exact(self, y_obs):
        """
        Pure LEDH without Log-transform. Daum(11)
        Linearizes h(x) at EACH particle location.
        """
        R = tf.constant([[1]], dtype=tf.float32)
        lam = 0

        for s in range(self.steps):
            P = self.compute_covariance(self.particles)
            H_i = 0.5 * self.model.beta * tf.exp(self.particles/2)
            H_i = tf.expand_dims(H_i, axis=1)
            h_val = self.model.beta * tf.exp(self.particles/2)
            P_exp = tf.reshape(P, [1, 1, 1])
            HPH = H_i * P_exp * H_i
            M = lam * HPH + R
            M_inv = 1/M
            PHt = P_exp * H_i
            A = -0.5 * PHt * M_inv * H_i
            Hx = H_i * tf.expand_dims(self.particles, -1)
            e = tf.expand_dims(h_val, -1) - Hx
            innov = y_obs - e
            R_inv = 1/R
            PHt_Rinv = PHt * R_inv
            term_innov = PHt_Rinv * innov
            I = 1
            term1 = (I + lam * A) * term_innov
            term2 = A * tf.expand_dims(self.particles, -1)
            b = (I + 2 * lam * A) * (term1 + term2)
            drift = A * tf.expand_dims(self.particles, -1) + b
            drift = tf.squeeze(drift, axis=-1)
            self.particles += self.dt * drift
            lam += self.dt
            # Diagnostics
            # A is [N, 1, 1], J is [N, 1, 1]
            J = tf.eye(1, batch_shape=[self.N]) + self.dt * A
            self._record_diagnostics(drift, J)

    def flow_edh_ledh_log_trans(self, y_obs):
        """
        EDH/LEDH with log(y^2) transform. (Li(17))
        """

        z_obs_trans = tf.math.log(y_obs**2 + 1e-8)
        H = tf.ones((1, 1), dtype=tf.float32)
        R = tf.constant([[self.obs_noise_var]], dtype=tf.float32)

        lam = 0
        for s in range(self.steps):
            P = self.compute_covariance(self.particles)
            HPH = tf.matmul(H, tf.matmul(P, H, transpose_b=True))
            inv_term = tf.linalg.inv(lam * HPH + R)
            PHt = tf.matmul(P, H, transpose_b=True)
            A = -0.5 * tf.matmul(PHt, tf.matmul(inv_term, H))
            bias = tf.math.log(self.model.beta**2) + self.obs_noise_mean
            innovation = z_obs_trans - bias
            K_gain = tf.matmul(PHt, tf.linalg.inv(R))
            I = tf.eye(1)
            term1 = tf.matmul(I + lam * A, K_gain) * innovation
            b = tf.matmul(I + 2 * lam * A, term1)
            drift = tf.matmul(self.particles, A, transpose_b=True) + b
            self.particles += self.dt * drift
            lam += self.dt
            # Diagnostics
            J = I + self.dt * A
            self._record_diagnostics(drift, J)

    def rbf_kernel(self, X):
        """
        Computes RBF Kernel matrix and gradient
        """

        # X: [N, 1]
        # dists: [N, N, 1]
        dists = X[:, None] - X[None, :]
        # Heuristic bandwidth (median trick approximation)
        h = tf.reduce_mean(tf.abs(dists)) + 1e-6
        # sq_dists: [N, N, 1]
        sq_dists = tf.square(dists)
        # K: [N, N, 1] -> Squeeze to [N, N] for matrix multiplication
        K = tf.exp(-sq_dists/(2*h**2))
        K = tf.squeeze(K, axis=-1)
        # grad_K: [N, N, 1] -> Squeeze to [N, N]
        grad_K = -(dists/(h**2))*K[:, :, None]
        grad_K = tf.squeeze(grad_K, axis=-1)

        return K, grad_K

    def flow_kernel(self, y_obs):
        """Kernel PFF (Hu21 Algorithm 1)"""
        for s in range(self.steps):
            with tf.GradientTape() as tape:
                tape.watch(self.particles)
                log_p = self.model.log_likelihood(y_obs, self.particles)

            grad_log_p = tape.gradient(log_p, self.particles)
            # K: [N, N], grad_K: [N, N]
            K, grad_K = self.rbf_kernel(self.particles)
            # Hu21 Eq 4 approximation
            # term1: K * grad_log_p -> [N, N] @ [N, 1] = [N, 1]
            term1 = tf.matmul(K, grad_log_p)
            # term2: sum_j grad_K(x_i, x_j) -> sum over columns -> [N, 1]
            term2 = tf.reduce_sum(grad_K, axis=1, keepdims=True)
            drift = (term1 + term2) / float(self.N)
            self.particles += self.dt * drift

            # Diagnostics (Kernel Jacobian is implicit/complex, using 1.0 placeholder)
            self._record_diagnostics(drift, None)


    def update(self, y_obs):
        if self.method == 'EDH_Log':
            self.flow_edh_ledh_log_trans(y_obs)
        elif self.method == 'EDH_Exact':
            self.flow_edh_exact(y_obs)
        elif self.method == 'LEDH_Exact':
            self.flow_ledh_exact(y_obs)
        elif self.method == 'Kernel':
            self.flow_kernel(y_obs)


class AcousticPFPF:
    """
    Particle Flow Particle Filter for Acoustic Tracking (Li (17))
    """

    def __init__(self, config, model, mode='LEDH'):
        self.cfg = config
        self.model = model
        self.mode = mode # 'LEDH' or 'EDH'
        # Initialize
        self.particles = self.cfg.x0_mean + tf.random.normal(
            [self.cfg.n_particles, self.cfg.state_dim]) @ tf.linalg.cholesky(self.cfg.P0).numpy().T
        self.weights = tf.ones([self.cfg.n_particles]) / self.cfg.n_particles
        # Stepsize according to C. Implementation and Complexity in Li(17)
        n_lambda = 29
        lambdas = np.zeros(n_lambda + 1)
        if n_lambda > 0:
            steps = []
            cur = 0.001 # initial step
            ratio = 1.2
            total = 0
            for i in range(n_lambda):
                steps.append(cur)
                total += cur
                cur *= ratio
            steps = np.array(steps) / total
            curr_lambda = 0
            for i in range(n_lambda):
                curr_lambda += steps[i]
                lambdas[i + 1] = curr_lambda
            lambdas[-1] = 1
        self.lambda_steps = tf.constant(lambdas, dtype=tf.float32)

    def compute_log_prob(self, x, mean, cov_inv):
        """
        calculate log prob
        term = (x-mu)^T Sigma^-1 (x-mu)
        diff: [batch, state_dim], cov_inv: [state_dim, state_dim]
        """
        diff = x - mean
        term = tf.reduce_sum(tf.matmul(diff, cov_inv) * diff, axis=1)
        return -0.5 * term

    def run_step(self, z_obs, P_pred):
        """
        run the filter for one step
        z_obs: [meas_dim]
        P_pred: predicted covariance by EKF [state_dim, state_dim]
        """
        # 1. Prediction (Sampling from prior/transition)
        # p(eta0 | x_k-1)
        particles_prev = self.particles
        # eta_0 = f(x_k-1) + v_k
        self.particles = self.model.transition(self.particles, noise=True)
        eta_0 = self.particles
        # 2. Particle Flow
        eta_1, log_det_jacobian_sum = self.run_flow(eta_0, z_obs, P_pred)
        # 3. Invertible Mapping Weight Update
        # log w_k = log w_k-1 + log p(z|eta_1) + log p(eta_1|x_k-1) - log p(eta_0|x_k-1) + log |det|
        # p(z|eta_1)
        z_pred_1 = self.model.measurement(eta_1)
        diff_z = tf.expand_dims(z_obs, 0) - z_pred_1
        log_like = -0.5 * tf.reduce_sum(tf.matmul(diff_z, self.cfg.R_inv) * diff_z, axis=1)
        # p(eta|x_k-1)
        mu_trans = self.model.transition(particles_prev, noise=False)
        log_trans_eta1 = self.compute_log_prob(eta_1, mu_trans, self.cfg.Q_inv)
        log_trans_eta0 = self.compute_log_prob(eta_0, mu_trans, self.cfg.Q_inv)

        log_weights = tf.math.log(self.weights + 1e-30)
        log_weights_new = log_weights + log_like + log_trans_eta1 - log_trans_eta0 + log_det_jacobian_sum
        # normalization
        max_log = tf.reduce_max(log_weights_new)
        weights_unnorm = tf.exp(log_weights_new - max_log)
        self.weights = weights_unnorm / tf.reduce_sum(weights_unnorm)
        self.particles = eta_1
        # 4. Systematic Resampling
        n_eff = 1 / tf.reduce_sum(tf.square(self.weights))
        if n_eff < self.cfg.n_particles / 2.0:
            indices = self.systematic_resample(self.weights)
            self.particles = tf.gather(self.particles, indices)
            self.weights = tf.ones([self.cfg.n_particles]) / self.cfg.n_particles
        # 5. state estimation
        x_est = tf.reduce_sum(self.particles * tf.expand_dims(self.weights, 1), axis=0)
        return x_est

    def run_flow(self, eta, z_obs, P):
        """
        Run Algorithm 1 (LEDH) or Algorithm 2 (EDH)
        """
        eta_curr = eta
        log_det_sum = tf.zeros([self.cfg.n_particles])
        if self.mode == 'LEDH':
            eta_aux = eta
        else:
            eta_aux = tf.reduce_mean(eta, axis=0, keepdims=True) # [1, state_dim]

        for i in range(len(self.lambda_steps) - 1):
            lam = self.lambda_steps[i + 1]
            dl = lam - self.lambda_steps[i]
            # calculate A, b
            linearize_at = eta_aux
            A, b = self.compute_flow_params(linearize_at, z_obs, P, lam)
            # Flow of auxiliary variable
            if self.mode == 'EDH':
                # A: [1, state_dim, state_dim], b: [1, state_dim] (fixed for every i)
                drift_aux = tf.squeeze(tf.matmul(eta_aux, A, transpose_b=True) + b, 0) # [state_dim]
                eta_aux = eta_aux + dl * drift_aux
                # Algorithm 2 Line 16
                drift = tf.matmul(eta_curr, A[0], transpose_b=True) + b
                eta_curr = eta_curr + dl * drift
                # Jacobian Determinant: det(I + dl * A)
                I = tf.eye(self.cfg.state_dim)
                det = tf.linalg.det(I + dl * A[0])
                log_det_sum += tf.math.log(tf.abs(det) + 1e-8)
            else: # LEDH
                # A: [batch, state_dim, state_dim], b: [batch, state_dim]
                # Algorithm 1 Line 17
                drift_aux = tf.einsum('nij,nj->ni', A, eta_aux) + b
                eta_aux = eta_aux + dl * drift_aux
                # Algorithm 1 Line 18
                drift = tf.einsum('nij,nj->ni', A, eta_curr) + b
                eta_curr = eta_curr + dl * drift
                # Jacobian Determinant: det(I + dl * A_i)
                I = tf.eye(self.cfg.state_dim)
                # I expanded: [1, state_dim, state_dim] broadcast to [batch, state_dim, state_dim]
                mat = tf.expand_dims(I, 0) + dl * A
                dets = tf.linalg.det(mat)
                log_det_sum += tf.math.log(tf.abs(dets) + 1e-30)
        return eta_curr, log_det_sum

    def compute_flow_params(self, eta_linearize, z, P, lam):
        """
        Calculate A(lambda) and b(lambda).
        eta_linearize: [batch, state_dim] (LEDH) or [1, state_dim] (EDH)
        z: [S], P: [state_dim, state_dim]
        """

        # 1. calculate H(lambda) at eta_linearize
        H = self.model.measurement_jacobian(eta_linearize)
        # 2. calculate M = lambda * H P H^T + R
        # P: [state_dim, state_dim] -> [1, state_dim, state_dim]
        P_exp = tf.expand_dims(P, 0)
        HP = tf.matmul(H, P_exp) # [batch, S, state_dim]
        HPHt = tf.matmul(HP, H, transpose_b=True) # [batch, S, S]
        R_exp = tf.expand_dims(self.cfg.R_cov, 0)
        M = lam * HPHt + R_exp # [batch, S, S]
        M_inv = tf.linalg.inv(M) # [batch, S, S]
        # 3. calculate A = -0.5 * P H^T M^-1 H (Eq 10/13)
        # term1: P H^T -> [batch, state_dim, S]
        PHt = tf.matmul(P_exp, H, transpose_b=True)
        # term2: M^-1 H -> [batch, S, state_dim]
        MinvH = tf.matmul(M_inv, H)
        A = -0.5 * tf.matmul(PHt, MinvH) # [batch, state_dim, state_dim]
        # 4. calculate b (Eq 11/14)
        # b = (I + 2*lam*A) [ (I + lam*A) P H^T R^-1 (z - e) + A * eta_bar ]
        # e = h(eta) - H * eta
        h_val = self.model.measurement(eta_linearize) # [batch, S]
        # H * eta -> [batch, S]
        H_eta = tf.einsum('nij,nj->ni', H, eta_linearize)
        e = h_val - H_eta
        # innov = z - e
        innov = tf.expand_dims(z, 0) - e # [batch, S]
        # term_inner_1 = P H^T R^-1 (z - e)
        # R^-1 (z-e) -> [batch, S]
        Rinv_innov = tf.einsum('ij,nj->ni', self.cfg.R_inv, innov)
        # PHt * ... -> [N, state_dim]
        term_inner_1 = tf.einsum('nij,nj->ni', PHt, Rinv_innov)
        # Apply (I + lam*A)
        # (I + lam*A) v = v + lam * (A*v)
        A_term1 = tf.einsum('nij,nj->ni', A, term_inner_1)
        part1 = term_inner_1 + lam * A_term1
        # term_inner_2 = A * eta_bar
        part2 = tf.einsum('nij,nj->ni', A, eta_linearize)
        bracket = part1 + part2
        # Final multiply by (I + 2*lam*A)
        A_bracket = tf.einsum('nij,nj->ni', A, bracket)
        b = bracket + 2 * lam * A_bracket
        return A, b

    def systematic_resample(self, weights):
        N = self.cfg.n_particles
        positions = (tf.range(N, dtype=tf.float32) + tf.random.uniform([], 0, 1)) / float(N)
        cumulative_sum = tf.cumsum(weights)
        indices = tf.searchsorted(cumulative_sum, positions)
        return tf.clip_by_value(indices, 0, N - 1)