import tensorflow as tf

class KernelEmbeddedPFF:
    """
    Kernel-Embedded Particle Flow Filter for high-dimensional systems (Hu(21)).
    """

    def __init__(self, nx, n_particles, obs_indices, R_std, kernel_type='matrix'):
        self.nx = nx
        self.Np = n_particles
        self.obs_indices = tf.constant(obs_indices, dtype=tf.int32)
        self.ny = len(obs_indices)
        self.R_inv = tf.eye(self.ny)*(1/(R_std**2))
        self.kernel_type = kernel_type
        self.alpha = 1/self.Np # width of the kernel P6 in Hu(21)

    def log_likelihood_grad(self, particles, y_obs):
        """
        Calculates gradient of log p(y|x).
        Assumes Linear Observation: H is selecting indices.
        grad = H^T R^-1 (y - Hx)
        input: particles: [Np,Nx], y_obs: [Ny,]
        return: [Np, Nx]
        """

        # Extract observed variables from particles: [Np, Ny]
        Hx = tf.gather(particles, self.obs_indices, axis=1)

        # Innovation: [Np, Ny]
        innov = y_obs - Hx

        # innov*R_inv: [Np, Ny]
        grad_obs = tf.matmul(innov, self.R_inv)

        # Map back to full state space [Np, Nx]
        indices = tf.expand_dims(self.obs_indices, 1)  # [Ny, 1]
        grads = []
        for i in range(self.Np):
            full_grad = tf.scatter_nd(indices, grad_obs[i], shape=[self.nx])
            grads.append(full_grad)

        return tf.stack(grads)  # [Np, Nx]

    def log_prior_grad(self, particles, prior_mean, B_inv):
        """
        Gradient of log p(x). Gaussian assumption.
        grad = -B^-1 (x - x_b)
        input: particles: [Np,Nx], prior_mean: [Nx], B_inv: [Nx,Nx]
        """
        diff = particles - prior_mean
        return -tf.matmul(diff, B_inv)

    def compute_kernel_matrix(self, particles, B_diag):
        """
        Computes K(x_i, x_j) and its gradient.
        Input:
            particles: [Np, Nx]
            B_diag: [Nx] (Diagonal of Prior Covariance)
        """

        # xi: [Np, 1, Nx], xj: [1, Np, Nx]
        xi = tf.expand_dims(particles, 1)
        xj = tf.expand_dims(particles, 0)
        # Normalized difference squared: (xi - xj)^2 / (alpha * sigma^2)
        scale = self.alpha * B_diag # [Nx]
        diff_sq = tf.square(xi - xj) # [Np, Np, Nx]
        arg = -0.5 * diff_sq / scale # [Np, Np, Nx]

        if self.kernel_type == 'scalar':
            # Scalar Kernel: K(x, z) = exp(-0.5 * (x-z)^T A (x-z))
            arg_sum = tf.reduce_sum(arg, axis=2) # [Np, Np]
            K_matrix = tf.exp(arg_sum) # [Np, Np]
            K_val = tf.expand_dims(K_matrix, 2) # [Np, Np, 1]
            # Gradient of Kernel: div_x K(xi, x)=-A^T (xi - x)*K (19)
            diff = (xj - xi)
            grad_K = (diff / scale) * K_val

        elif self.kernel_type == 'matrix':
            # Matrix-valued Kernel: Diagonal matrix
            # (20) & (21): K is computed per component independently
            K_val = tf.exp(arg) # [Np, Np, Nx]
            # Gradient of Kernel (23)
            diff = (xj - xi)
            grad_K = (diff / scale) * K_val

        return K_val, grad_K

    def update(self, particles, y_obs, n_steps=50, dt_s=0.05):
        """
        Performs the particle flow update (pseudo-time integration).
        """

        curr_particles = particles

        # Prior Statistics
        prior_mean = tf.reduce_mean(curr_particles, axis=0)
        perturbations = curr_particles - prior_mean
        B_diag = tf.reduce_mean(tf.square(perturbations), axis=0)
        B_diag = tf.maximum(B_diag, 1e-6) # Guarantee PD
        D = B_diag # Localized prior covariance matrix

        for s in range(n_steps):
            grad_lik = self.log_likelihood_grad(curr_particles, y_obs)
            grad_prior = self.log_prior_grad(curr_particles, prior_mean, tf.linalg.diag(1.0 / B_diag))
            grad_log_p = grad_lik + grad_prior # [Np, Nx]

            # K_val: [Np, Np, Nx] (if matrix) or [Np, Np, 1] (if scalar)
            # grad_K: [Np, Np, Nx]
            K_val, grad_K = self.compute_kernel_matrix(curr_particles, B_diag)

            # Flow calculation ((6) in Hu(21))
            grad_p_expanded = tf.expand_dims(grad_log_p, 1) #[Np, 1, Nx]
            term1_inner = K_val * grad_p_expanded # [Np, Np, Nx]
            term2_inner = grad_K #[Np, Np, Nx]

            integral_sum = tf.reduce_mean(term1_inner + term2_inner, axis=0) # [Np, Nx]
            flow = D * integral_sum # [Np, Nx]

            curr_particles = curr_particles + dt_s * flow

        return curr_particles