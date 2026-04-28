import pytest
import tensorflow as tf
import numpy as np
from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel


class TestNonlinearSSMModel:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Initialize config and model
        self.cfg = NonlinearSSMConfig(T=100)
        self.model = NonlinearSSMModel(self.cfg)
        self.rng = tf.random.Generator.from_seed(42)

    def test_config_initialization(self):
        """Test config class initialization"""
        assert self.cfg.state_dim == 1
        assert self.cfg.meas_dim == 1
        assert self.cfg.T == 100
        assert np.isclose(self.cfg.sigma_v2, 10.0)
        assert np.isclose(self.cfg.sigma_w2, 1.0)

    def test_sample_initial_particles(self):
        """Test initial particle sampling"""
        n_particles = 1000
        particles = self.model.sample_initial_particles(n_particles)
        assert particles.shape == (n_particles, self.cfg.state_dim)
        # Verify initial distribution mean and variance
        mean = tf.reduce_mean(particles).numpy()
        var = tf.math.reduce_variance(particles).numpy()
        assert np.isclose(mean, 0.0, atol=0.5)
        assert np.isclose(var, 5.0, atol=1.0)

    def test_transition_mean(self):
        """Test state transition mean calculation"""
        # Test scalar input
        x = tf.constant([0.0])
        t = 1
        x_mean = self.model.transition_mean(x, t)
        # When x=0, transition mean=0 + 0 + 8*cos(0) = 8.0
        assert x_mean.shape == (1,)
        assert np.isclose(x_mean.numpy()[0], 8.0)

        # Test batch input
        x_batch = tf.constant([[0.0], [1.0], [2.0]])
        x_mean_batch = self.model.transition_mean(x_batch, t=1)
        assert x_mean_batch.shape == (3, 1)

    def test_propagate_particles(self):
        """Test particle propagation"""
        n_particles = 100
        x_prev = self.model.sample_initial_particles(n_particles)
        x_pred = self.model.propagate_particles(x_prev, t=1)
        assert x_pred.shape == (n_particles, self.cfg.state_dim)

    def test_measurement_model(self):
        """Test measurement model"""
        x = tf.constant([10.0])
        z_mean = self.model.measurement_model(x)
        # 0.05*(10^2) = 5.0
        assert z_mean.shape == (1,)
        assert np.isclose(z_mean.numpy()[0], 5.0)

        x_batch = tf.constant([[0.0], [5.0], [10.0]])
        z_mean_batch = self.model.measurement_model(x_batch)
        assert z_mean_batch.shape == (3, 1)
        assert np.isclose(z_mean_batch.numpy()[2, 0], 5.0)

    def test_measurement_jacobian(self):
        """Test measurement Jacobian matrix"""
        x = tf.constant([10.0])
        H = self.model.measurement_jacobian(x)
        # dh/dx = 2*0.05*10 = 1.0
        assert H.shape == (1, 1)
        assert np.isclose(H.numpy()[0, 0], 1.0)

        x_batch = tf.constant([[0.0], [5.0], [10.0]])
        H_batch = self.model.measurement_jacobian(x_batch)
        assert H_batch.shape == (3, 1, 1)
        assert np.isclose(H_batch.numpy()[2, 0, 0], 1.0)

    def test_transition_log_prob(self):
        """Test transition log probability"""
        x_prev = tf.constant([[0.0]])
        x_next = tf.constant([[8.0]])
        t = 1
        log_prob = self.model.transition_log_prob(x_next, x_prev, t)
        assert log_prob.shape == (1,)
        # Mean is 8, x_next=8, log probability is value of N(8,10) at 8
        true_log_prob = tf.math.log(1.0 / tf.sqrt(2 * np.pi * 10.0))
        assert np.isclose(log_prob.numpy()[0], true_log_prob.numpy())

    def test_measurement_log_prob(self):
        """Test measurement log probability"""
        x = tf.constant([[10.0]])
        z = tf.constant([5.0])
        log_prob = self.model.measurement_log_prob(z, x)
        assert log_prob.shape == (1,)
        true_log_prob = tf.math.log(1.0 / tf.sqrt(2 * np.pi * 1.0))
        assert np.isclose(log_prob.numpy()[0], true_log_prob.numpy())

    def test_generate_true_trajectory(self):
        """Test true trajectory generation"""
        x_true, y_obs = self.model.generate_true_trajectory()
        assert x_true.shape == (self.cfg.T, self.cfg.state_dim)
        assert y_obs.shape == (self.cfg.T, self.cfg.meas_dim)