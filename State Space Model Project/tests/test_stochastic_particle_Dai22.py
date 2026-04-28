import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from src.filters.stochastic_particle_flow_base import BaseParticleFlowFilterCore
from src.filters.stochastic_particle_flow_Dai22 import ParticleFlowFilterDai22
from src.models.stochastic_particle_EX1 import Dai22Example1Model


# Set random seed for reproducibility
@pytest.fixture(autouse=True)
def set_random_seed():
    tf.random.set_seed(1234)
    np.random.seed(1234)


# Fixture for the test model
@pytest.fixture
def model():
    return Dai22Example1Model()


# Fixture for standard filter instance
@pytest.fixture
def filter_instance(model):
    return ParticleFlowFilterDai22(
        model=model,
        num_particles=30,
        num_steps=20,
        resample=False,
        diffusion_scale=0.03,
        jitter=1e-4,
        drift_clip=1.5,
    )


# Fixture for filter with resampling enabled
@pytest.fixture
def filter_resample(model):
    return ParticleFlowFilterDai22(
        model=model,
        num_particles=30,
        num_steps=20,
        resample=True,
        diffusion_scale=0.03,
        jitter=1e-4,
        drift_clip=1.5,
    )


# Fixture for prior particles sampled from model
@pytest.fixture
def prior_particles(model, filter_instance):
    return model.sample_initial_particles(filter_instance.N)


# Fixture for test measurement vector
@pytest.fixture
def test_measurement():
    return tf.constant([0.4754, 1.1868], dtype=tf.float32)


# Dummy model with boundary constraints for testing
class DummyBoundaryModel(Dai22Example1Model):
    def reflect_boundaries(self, particles):
        return tf.clip_by_value(particles, -1.0, 1.0)


# Dummy model for autodiff Jacobian testing
class DummyAutoDiffModel:
    def __init__(self):
        self.prior_mean = tf.constant([0.5, -0.5], dtype=tf.float32)
        self.prior_cov = tf.eye(2, dtype=tf.float32)
        self.R = 0.1 * tf.eye(2, dtype=tf.float32)
        self.Q_flow = 0.01 * tf.eye(2, dtype=tf.float32)
        self.mu = 0.1

    def measurement_model(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.stack([x[..., 0] ** 2, x[..., 1] + 1.0], axis=-1)

    def get_measurement_cov(self, x=None):
        return self.R

    def sample_initial_particles(self, num_particles):
        return tf.random.normal((num_particles, 2), dtype=tf.float32)

    def propagate_particles(self, particles):
        noise = 0.01 * tf.random.normal(tf.shape(particles), dtype=tf.float32)
        return particles + noise


# Model Initialization Tests
def test_model_initialization(model):
    """Verify model parameters and dimensions are correctly initialized"""
    assert model.state_dim == 2
    assert model.meas_dim == 2
    assert tf.equal(model.x_true, [4.0, 4.0]).numpy().all()
    assert model.prior_mean.shape == (2,)
    assert model.prior_cov.shape == (2, 2)


def test_model_measurement_function(model):
    """Test bearing measurement model outputs valid values"""
    x = tf.constant([4.0, 4.0], dtype=tf.float32)
    measurement = model.measurement_model(x)
    assert measurement.shape == (2,)
    assert tf.math.is_finite(measurement).numpy().all()


def test_model_propagation(model, prior_particles):
    """Test static particle propagation (identity function)"""
    propagated = model.propagate_particles(prior_particles)
    np.testing.assert_allclose(propagated.numpy(), prior_particles.numpy())


def test_model_prior_distribution(model):
    """Test prior distribution sampling"""
    dist = model.get_prior_dist()
    sample = dist.sample()
    assert sample.shape == (2,)


def test_model_measurement_sampling(model):
    """Test noisy measurement generation"""
    z = model.sample_measurement()
    assert z.shape == (2,)
    assert tf.math.is_finite(z).numpy().all()


# Filter Initialization Tests
def test_filter_initialization(model):
    """Verify filter attributes are set correctly"""
    pf = ParticleFlowFilterDai22(
        model=model,
        num_particles=40,
        num_steps=25,
        resample=True,
        diffusion_scale=0.05,
        jitter=1e-3,
        drift_clip=2.0,
    )
    assert pf.model == model
    assert pf.N == 40
    assert pf.num_steps == 25
    assert np.isclose(pf.dlambda, 1.0 / 25)
    assert pf.resample is True


# Base Filter Utility Tests
def test_sample_covariance(filter_instance, prior_particles):
    """Test sample mean and covariance calculation"""
    mean, cov = filter_instance._sample_covariance(prior_particles)
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)
    assert np.isfinite(mean.numpy()).all()
    assert np.isfinite(cov.numpy()).all()


def test_matrix_symmetrization(filter_instance):
    """Test matrix symmetrization utility function"""
    A = tf.constant([[1.0, 2.0], [0.0, 3.0]], dtype=tf.float32)
    sym = filter_instance._symmetrize(A)
    expected = np.array([[1.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(sym.numpy(), expected)


def test_matrix_regularization(filter_instance):
    """Test matrix regularization with jitter"""
    A = tf.constant([[1.0, 2.0], [0.0, 3.0]], dtype=tf.float32)
    reg = filter_instance._regularize_matrix(A, eps=1e-2)
    assert np.allclose(reg.numpy(), reg.numpy().T)


def test_safe_matrix_inverse(filter_instance):
    """Test stable matrix inversion"""
    A = tf.constant([[2.0, 0.0], [0.0, 4.0]], dtype=tf.float32)
    inv = filter_instance._safe_inverse(A)
    identity = inv @ A
    np.testing.assert_allclose(identity.numpy(), np.eye(2), atol=1e-3)


def test_boundary_application():
    """Test boundary reflection for constrained models"""
    model = DummyBoundaryModel()
    pf = ParticleFlowFilterDai22(model=model, num_particles=5, num_steps=5)
    particles = tf.constant([[2.0, -3.0], [0.5, 0.2]], dtype=tf.float32)
    bounded = pf._apply_boundaries(particles)
    assert bounded.shape == particles.shape


def test_measurement_evaluation(filter_instance):
    """Test measurement model wrapper function"""
    x = tf.constant([[4.0, 4.0], [3.0, 5.0]], dtype=tf.float32)
    z = filter_instance._measurement_eval(x)
    assert z.shape == (2, 2)


def test_diffusion_covariance(filter_instance):
    """Test flow diffusion covariance matrix generation"""
    Q = filter_instance._get_diffusion_cov()
    assert Q.shape == (2, 2)
    eigvals = np.linalg.eigvals(Q.numpy())
    assert (eigvals > 0).all()


def test_linear_beta_schedule(filter_instance):
    """Test baseline linear homotopy schedule"""
    beta, beta_dot = filter_instance._linear_beta_schedule()
    assert len(beta) == filter_instance.num_steps + 1
    assert beta[0] == 0.0
    assert beta[-1] == 1.0


# Homotopy & Hessian Tests
def test_hessian_calculation(filter_instance, model, test_measurement):
    """Test Hessian computation for prior and likelihood"""
    H0, Hh = filter_instance._compute_hessians(test_measurement, model.prior_mean, model.prior_cov)
    assert H0.shape == (2, 2)
    assert Hh.shape == (2, 2)


def test_optimal_beta_solver(filter_instance, model, test_measurement):
    """Test optimal homotopy beta(lambda) solver"""
    beta, beta_dot = filter_instance.solve_optimal_beta(test_measurement, model.prior_mean, model.prior_cov)
    assert len(beta) == filter_instance.num_steps + 1
    assert np.isclose(beta[0], 0.0)
    assert np.isclose(beta[-1], 1.0)

# Particle Flow Tests
def test_particle_flow_execution(filter_instance, model, prior_particles, test_measurement):
    """Test full stochastic particle flow execution"""
    beta, beta_dot = filter_instance.solve_optimal_beta(test_measurement, model.prior_mean, model.prior_cov)
    particles = filter_instance.run_flow(
        z=test_measurement,
        beta_vals=beta,
        beta_dot_vals=beta_dot,
        prior_particles=prior_particles,
        prior_mean=model.prior_mean,
        prior_cov=model.prior_cov
    )
    assert particles.shape == prior_particles.shape
    assert tf.math.is_finite(particles).numpy().all()


def test_stiffness_ratio_calculation(filter_instance, model, test_measurement):
    """Test stiffness ratio computation along homotopy path"""
    beta, _ = filter_instance.solve_optimal_beta(test_measurement, model.prior_mean, model.prior_cov)
    stiffness = filter_instance.compute_stiffness_ratio_path(
        z=test_measurement,
        beta_vals=beta,
        prior_mean=model.prior_mean,
        prior_cov=model.prior_cov
    )
    assert stiffness.shape == (filter_instance.num_steps + 1,)
    assert (stiffness.numpy() >= 0).all()


# Filtering Step Tests
def test_prediction_step(filter_instance, prior_particles):
    """Test particle prediction step"""
    predicted = filter_instance.predict(prior_particles)
    assert predicted.shape == prior_particles.shape


def test_update_step(filter_instance, prior_particles, test_measurement):
    """Test single filter update step"""
    updated, mean, cov, beta, bd = filter_instance.update(prior_particles, test_measurement)
    assert updated.shape == prior_particles.shape
    assert mean.shape == (2,)


def test_update_with_resampling(filter_resample, prior_particles, test_measurement):
    """Test update step with resampling enabled"""
    updated, mean, cov, beta, bd = filter_resample.update(prior_particles, test_measurement)
    assert updated.shape == prior_particles.shape
    assert tf.math.is_finite(updated).numpy().all()


def test_filter_step(filter_instance, prior_particles, test_measurement):
    """Test complete filtering step (predict + update)"""
    particles, info = filter_instance.filter_step(prior_particles, test_measurement)
    assert "estimate" in info
    assert info["estimate"].shape == (2,)


def test_filter_sequence(filter_instance, prior_particles):
    """Test sequential filtering over multiple measurements"""
    measurements = tf.constant([[0.4754, 1.1868], [0.5000, 1.1500]], dtype=tf.float32)
    seq, info, estimates, cov = filter_instance.filter_sequence(prior_particles, measurements)
    assert len(seq) == 2
    assert estimates.shape == (2, 2)


# Integrated Pipeline Test
def test_full_filter_pipeline(model):
    """End-to-end test of the full particle flow pipeline"""
    pf = ParticleFlowFilterDai22(model=model, num_particles=50, num_steps=50)
    z = tf.constant([0.4754, 1.1868], dtype=tf.float32)
    prior = model.sample_initial_particles(pf.N)

    beta, beta_dot = pf.solve_optimal_beta(z, model.prior_mean, model.prior_cov)
    particles = pf.run_flow(z, beta, beta_dot, prior, model.prior_mean, model.prior_cov)

    mean = tf.reduce_mean(particles, axis=0)
    assert tf.math.is_finite(mean).numpy().all()



# Main Experiment Smoke Tests
def test_static_example():
    """Smoke test for static example from main script"""
    from main_stochastic_Dai22 import run_static_example
    results = run_static_example()
    assert np.isfinite(results[1].numpy()).all()


def test_monte_carlo_experiment():
    """Smoke test for Monte Carlo table experiment"""
    from main_stochastic_Dai22 import run_monte_carlo_table1
    results = run_monte_carlo_table1(num_mc_runs=1, num_particles=10, num_steps=5)
    assert "average" in results


