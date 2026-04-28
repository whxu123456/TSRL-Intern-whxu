import tensorflow as tf

class Lorenz96:
    """
    Simulation of Lorenz 96 (Hu(21)).
    """

    def __init__(self, nx=1000, F=8):
        self.nx = nx
        self.F = F

    def dynamics(self, x):
        """
        dx/dt = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        input:
        x: (nx, )
        return:
        dxdt: (nx, )
        """

        x_m2 = tf.roll(x, shift=2, axis=-1)
        x_m1 = tf.roll(x, shift=1, axis=-1)
        x_p1 = tf.roll(x, shift=-1, axis=-1)
        dxdt = (x_p1 - x_m2) * x_m1 - x + self.F
        return dxdt

    def rk4_step(self, x, dt):
        """
        Fourth-order Runge-Kutta integration.
        input:
        x: (nx, )
        dt: const
        return: (nx, )
        """

        k1 = self.dynamics(x)
        k2 = self.dynamics(x + k1 * dt / 2)
        k3 = self.dynamics(x + k2 * dt / 2)
        k4 = self.dynamics(x + k3 * dt)
        return x + (k1 + 2*k2 + 2*k3 + k4) * dt / 6