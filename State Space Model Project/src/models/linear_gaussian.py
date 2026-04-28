import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class LinearGaussianSSM:
    """
    Linear Gaussian State Space Model.
    Ref: Example 2, Doucet (09).
    """

    def __init__(self, state_dim, process_noise_dim, obs_dim, obs_noise_dim,
                 A, B, C, D, Q, R, initial_mean, initial_cov):
        self.state_dim = state_dim # n_x
        self.process_noise_dim = process_noise_dim # n_v
        self.obs_dim = obs_dim # n_y
        self.obs_noise_dim = obs_noise_dim # n_w
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = Q #I_{n_v}
        self.R = R #I_{n_w}
        self.initial_mean = initial_mean
        self.initial_cov = initial_cov

        self.process_noise_cov = self.B @ self.Q @ tf.transpose(self.B)
        self.obs_noise_cov = self.D @ self.R @ tf.transpose(self.D)

    def generate_data(self, num_timesteps):
        """
        generate the LGSSM data
        return:
        states: [num_steps, state_dim]
        observations: [num_steps, obs_dim]
        """
        states = []
        observations = []

        # Initial state
        current_state = tfd.MultivariateNormalTriL(
            loc=self.initial_mean, scale_tril=tf.linalg.cholesky(self.initial_cov)
        ).sample()
        states.append(current_state)

        for t in range(num_timesteps):
            # Process update
            # X_n = A X_{n-1} + B V_n
            process_noise = tfd.MultivariateNormalDiag(
                loc=tf.zeros(self.process_noise_dim), scale_diag=tf.ones(self.process_noise_dim)
            ).sample()
            current_state = tf.linalg.matvec(self.A, current_state)+tf.linalg.matvec(self.B, process_noise)

            # Observation update
            # Y_n = C X_n + D W_n
            obs_noise = tfd.MultivariateNormalDiag(
                loc=tf.zeros(self.obs_dim), scale_diag=tf.ones(self.obs_noise_dim)
            ).sample()
            observation = tf.linalg.matvec(self.C, current_state)+tf.linalg.matvec(self.D, obs_noise)

            states.append(current_state)
            observations.append(observation)

        return tf.stack(states), tf.stack(observations)