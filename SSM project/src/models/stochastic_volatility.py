import numpy as np
import tensorflow as tf


class SVModel:
    """
    Stochastic Volatility Model.
    Ref: Example 4 in Doucet (09).
    """

    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5, T=100):
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.T = T

    def generate_data(self):
        """
        Generate stochastic volatility data.
        return:
        x,y: (T,)
        """
        x = np.zeros(self.T)
        y = np.zeros(self.T)

        # Initial state X1 ~ N(0, sigma^2/(1-alpha^2))
        x[0] = np.random.normal(0, self.sigma/np.sqrt(1-self.alpha**2))
        y[0] = self.beta*np.exp(x[0]/2)*np.random.normal(0, 1)

        for t in range(1, self.T):
            x[t] = self.alpha*x[t-1]+self.sigma*np.random.normal(0, 1)
            y[t] = self.beta*np.exp(x[t]/2)*np.random.normal(0, 1)

        return tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.float32)

    @tf.function
    def log_likelihood(self, y, x):
        """
        Computes log p(y|x) for the SV model.
        y ~ N(0, beta^2 * exp(x))
        """
        var = (self.beta**2)*tf.exp(x)
        var = tf.maximum(var,1e-6)
        log_prob = -0.5*tf.math.log(2*np.pi*var)-0.5*(y**2)/var
        return log_prob
