import numpy as np
import pytest
import tensorflow as tf

from src.models.acoustic import AcousticConfig, AcousticModel
from src.filters.EKF_acoustic import EKF_Predictor
from src.filters.acoustic_pfpf import AcousticPFPF
from src.filters.particle_flow_base import BaseParticleFlowFilter


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set global random seed for reproducibility"""
    tf.random.set_seed(42)
    np.random.seed(42)


@pytest.fixture
def cfg():
    """Fixture for AcousticConfig with reduced particles for fast testing"""
    config = AcousticConfig()
    config.n_particles = 50
    return config


@pytest.fixture
def model(cfg):
    """Fixture for AcousticModel instance"""
    return AcousticModel(cfg)


@pytest.fixture
def ekf(cfg, model):
    """Fixture for EKF_Predictor instance"""
    return EKF_Predictor(cfg, model)


@pytest.fixture
def p_pred(cfg):
    """Fixture for dummy predictive covariance matrix"""
    return tf.eye(cfg.state_dim, dtype=tf.float32) * 0.1


@pytest.fixture
def z_obs(cfg):
    """Fixture for dummy measurement vector"""
    return tf.zeros((cfg.meas_dim,), dtype=tf.float32)


@pytest.fixture(params=["LEDH", "EDH"])
def pf(request, cfg, model):
    """Fixture for AcousticPFPF (test both LEDH and EDH modes)"""
    return AcousticPFPF(cfg, model, mode=request.param)


def assert_all_finite(x):
    """Assert tensor/array has no NaN or Inf values"""
    arr = x.numpy() if tf.is_tensor(x) else np.asarray(x)
    assert np.all(np.isfinite(arr)), "Found NaN or Inf values in tensor/array."


# Test AcousticConfig
def test_config_core_dimensions(cfg):
    """Verify all core dimension parameters of AcousticConfig are correct"""
    assert cfg.n_targets == 4
    assert cfg.state_dim_per_target == 4
    assert cfg.state_dim == 16
    assert cfg.meas_dim == 25
    assert cfg.sensors_pos.shape == (25, 2)
    assert cfg.dt == 1.0
    assert cfg.T == 40


def test_config_covariance_matrices(cfg):
    """Verify shape and validity of all covariance matrices in config"""
    assert cfg.P0.shape == (cfg.state_dim, cfg.state_dim)
    assert cfg.Q_true.shape == (cfg.state_dim, cfg.state_dim)
    assert cfg.Q_cov.shape == (cfg.state_dim, cfg.state_dim)
    assert cfg.R_cov.shape == (cfg.meas_dim, cfg.meas_dim)
    assert cfg.F.shape == (cfg.state_dim, cfg.state_dim)
    assert_all_finite(cfg.P0)
    assert_all_finite(cfg.Q_cov)


def test_config_boundary_parameters(cfg):
    """Verify spatial boundary constraints are correctly set"""
    assert cfg.min_pos == 0.0
    assert cfg.max_pos == 40.0


# Test AcousticModel
def test_model_private_helper_functions(model, cfg):
    """Test private helper methods _ensure_batch and _maybe_squeeze"""
    # Test _ensure_batch
    x_single = tf.random.normal((cfg.state_dim,))
    x_batch = model._ensure_batch(x_single)
    assert x_batch.shape == (1, cfg.state_dim)

    # Test _maybe_squeeze
    x_squeezed = model._maybe_squeeze(x_batch, squeeze=True)
    assert x_squeezed.shape == (cfg.state_dim,)


def test_sample_initial_particles(model, cfg):
    """Test initial particle sampling returns valid shape and finite values"""
    particles = model.sample_initial_particles(cfg.n_particles)
    assert particles.shape == (cfg.n_particles, cfg.state_dim)
    assert_all_finite(particles)


@pytest.mark.parametrize("batch_size", [1, 5])
def test_transition_mean(model, cfg, batch_size):
    """Test transition mean function for batch and single state inputs"""
    x = tf.random.normal((batch_size, cfg.state_dim))
    x_next = model.transition_mean(x)
    assert x_next.shape == (batch_size, cfg.state_dim)
    assert_all_finite(x_next)


@pytest.mark.parametrize("noise", [True, False])
def test_transition_function(model, cfg, noise):
    """Test state transition with/without process noise"""
    x = tf.random.normal((3, cfg.state_dim))
    x_new = model.transition(x, noise=noise)
    assert x_new.shape == (3, cfg.state_dim)
    assert_all_finite(x_new)


def test_propagate_truth(model, cfg):
    """Test true state propagation with Q_true covariance"""
    x = tf.random.normal((3, cfg.state_dim))
    x_next = model.propagate_truth(x)
    assert x_next.shape == (3, cfg.state_dim)
    assert_all_finite(x_next)


def test_propagate_particles(model, cfg):
    """Test particle propagation with filter process noise"""
    particles = tf.random.normal((7, cfg.state_dim))
    x_next = model.propagate_particles(particles)
    assert x_next.shape == (7, cfg.state_dim)
    assert_all_finite(x_next)


@pytest.mark.parametrize("use_truth_noise", [True, False])
def test_transition_log_prob(model, cfg, use_truth_noise):
    """Test transition log probability with both noise models"""
    x_prev = tf.random.normal((4, cfg.state_dim))
    x_next = model.transition(x_prev, noise=False)
    logp = model.transition_log_prob(x_next, x_prev, use_truth_noise)
    assert logp.shape == (4,)
    assert_all_finite(logp)


def test_measurement_model(model, cfg):
    """Test deterministic measurement model (mean) for single/batch inputs"""
    x_batch = tf.random.normal((3, cfg.state_dim))
    x_single = tf.random.normal((cfg.state_dim,))

    z_batch = model.measurement_model(x_batch)
    z_single = model.measurement_model(x_single)

    assert z_batch.shape == (3, cfg.meas_dim)
    assert z_single.shape == (cfg.meas_dim,)
    assert_all_finite(z_batch)


def test_sample_measurement(model, cfg):
    """Test noisy measurement sampling adds valid noise"""
    x = tf.random.normal((5, cfg.state_dim))
    z_noisy = model.sample_measurement(x)
    assert z_noisy.shape == (5, cfg.meas_dim)
    assert_all_finite(z_noisy)


def test_measurement_covariance(model, cfg):
    """Test measurement covariance matrix returns correct shape"""
    R = model.get_measurement_cov()
    assert R.shape == (cfg.meas_dim, cfg.meas_dim)
    assert_all_finite(R)


def test_measurement_log_prob(model, cfg):
    """Test measurement log likelihood computation"""
    x = tf.random.normal((4, cfg.state_dim))
    z = model.sample_measurement(x)
    logp = model.measurement_log_prob(z, x)
    assert logp.shape == (4,)
    assert_all_finite(logp)


def test_measurement_jacobian(model, cfg):
    """Test analytical Jacobian matches autodiff results"""
    x = tf.random.normal((2, cfg.state_dim))
    H_analytic = model.measurement_jacobian(x)

    # Verify with TF GradientTape
    with tf.GradientTape() as tape:
        tape.watch(x)
        z = model.measurement_model(x)
    H_autodiff = tape.batch_jacobian(z, x)

    assert H_analytic.shape == (2, cfg.meas_dim, cfg.state_dim)
    np.testing.assert_allclose(H_analytic.numpy(), H_autodiff.numpy(), rtol=1e-3, atol=1e-3)


def test_reflect_boundaries(model, cfg):
    """Test boundary reflection flips velocity and clamps position"""
    # Create out-of-bounds state
    x_bad = np.zeros(cfg.state_dim, dtype=np.float32)
    x_bad[0] = -10.0  # Target 0 x (below min)
    x_bad[1] = 50.0  # Target 0 y (above max)
    x_bad[2] = 1.0  # Velocity x
    x_bad[3] = 1.0  # Velocity y

    x_reflect = model.reflect_boundaries(tf.constant(x_bad)).numpy()

    # Verify position is clamped and velocity is flipped
    assert 0.0 <= x_reflect[0] <= 40.0
    assert 0.0 <= x_reflect[1] <= 40.0
    assert x_reflect[2] == pytest.approx(-1.0)
    assert x_reflect[3] == pytest.approx(-1.0)


# Test EKF_Predictor
def test_ekf_initialization(ekf, cfg):
    """Verify EKF initial state and covariance are valid"""
    assert ekf.P.shape == (cfg.state_dim, cfg.state_dim)
    assert ekf.x.shape == (1, cfg.state_dim)
    assert_all_finite(ekf.P)
    assert_all_finite(ekf.x)


def test_ekf_predict_step(ekf, cfg):
    """Test EKF prediction step updates covariance and state"""
    P_old = tf.identity(ekf.P)
    P_pred = ekf.predict()

    assert P_pred.shape == (cfg.state_dim, cfg.state_dim)
    assert_all_finite(P_pred)
    # Covariance should increase after prediction
    assert np.all(tf.linalg.diag_part(P_pred) >= tf.linalg.diag_part(P_old))


def test_ekf_update_step(ekf, cfg, z_obs):
    """Test EKF update step corrects state and keeps covariance symmetric"""
    ekf.predict()
    x_pred = tf.identity(ekf.x)
    ekf.update(z_obs)

    # State should change after update
    assert not np.allclose(x_pred.numpy(), ekf.x.numpy())
    # Covariance must be symmetric
    np.testing.assert_allclose(ekf.P.numpy(), ekf.P.numpy().T, atol=1e-5)


def test_ekf_predict_update_stability(ekf, z_obs):
    """Test EKF predict+update pipeline produces no NaN/Inf values"""
    ekf.predict()
    ekf.update(z_obs)
    assert_all_finite(ekf.x)
    assert_all_finite(ekf.P)



# Test BaseParticleFlowFilter
def test_base_pf_compute_covariance(pf, cfg):
    """Test sample covariance computation of particle cloud"""
    cov = pf.compute_covariance(pf.particles)
    assert cov.shape == (cfg.state_dim, cfg.state_dim)
    assert_all_finite(cov)


def test_base_pf_weight_functions(pf):
    """Test log weight normalization and ESS calculation"""
    log_w = tf.math.log(tf.constant([0.2, 0.3, 0.5], dtype=tf.float32))
    w = pf._normalize_log_weights(log_w)

    assert np.all(w.numpy() >= 0.0)
    assert tf.reduce_sum(w).numpy() == pytest.approx(1.0)

    # Test ESS
    ess = pf.effective_sample_size(w)
    assert ess > 0


def test_base_pf_resampling(pf):
    """Test systematic resampling returns valid particle indices"""
    weights = tf.ones((pf.num_particles,), dtype=tf.float32) / pf.num_particles
    indices = pf.systematic_resample(weights)

    assert indices.shape == (pf.num_particles,)
    assert np.all(indices.numpy() >= 0)
    assert np.all(indices.numpy() < pf.num_particles)


def test_base_pf_diagnostics(pf):
    """Test flow stability diagnostics recording"""
    drift = tf.random.normal((pf.num_particles, pf.state_dim))
    jacobian = tf.eye(pf.state_dim, batch_shape=[pf.num_particles])
    pf._record_diagnostics(drift, jacobian)
    assert len(pf.flow_mags) > 0
    assert len(pf.condition_numbers) > 0


def test_base_pf_abstract_methods(pf):
    """Verify abstract methods raise NotImplementedError"""
    with pytest.raises(NotImplementedError):
        pf.predict()
    with pytest.raises(NotImplementedError):
        pf.update(tf.zeros(pf.meas_dim))


# Test AcousticPFPF
def test_pf_initialization(pf, cfg):
    """Verify PF-PF initial particles, weights, and lambda schedule"""
    assert pf.particles.shape == (cfg.n_particles, cfg.state_dim)
    assert pf.weights.shape == (cfg.n_particles,)
    assert tf.reduce_sum(pf.weights).numpy() == pytest.approx(1.0)
    assert_all_finite(pf.particles)


def test_pf_lambda_schedule(pf):
    """Test exponential lambda schedule is monotonic and valid"""
    lambdas = pf.lambda_steps.numpy()
    assert lambdas[0] == pytest.approx(0.0)
    assert lambdas[-1] == pytest.approx(1.0)
    assert np.all(np.diff(lambdas) >= 0.0)


def test_pf_flow_parameters(pf, z_obs, p_pred):
    """Test flow matrix A and bias b computation for EDH/LEDH"""
    eta = pf.particles if pf.mode == "LEDH" else tf.reduce_mean(pf.particles, axis=0, keepdims=True)
    A, b = pf.compute_flow_params(eta, z_obs, p_pred, lam=0.2)

    assert len(A.shape) == 3
    assert len(b.shape) == 2
    assert_all_finite(A)
    assert_all_finite(b)


def test_pf_flow_steps(pf, cfg):
    """Test EDH/LEDH flow step functions return valid particles"""
    x = pf.particles
    A = tf.eye(cfg.state_dim, batch_shape=[1 if pf.mode == "EDH" else cfg.n_particles]) * -0.01
    b = tf.zeros((A.shape[0], cfg.state_dim))

    if pf.mode == "EDH":
        out = pf._flow_step_edh(x, A, b, dl=0.1)
    else:
        out = pf._flow_step_ledh(x, A, b, dl=0.1)

    assert out.shape == x.shape
    assert_all_finite(out)


def test_pf_run_flow(pf, model, z_obs, p_pred):
    """Test full particle flow returns valid particles and log determinant"""
    eta0 = model.propagate_particles(pf.particles)
    eta1, log_det = pf.run_flow(eta0, z_obs, p_pred)

    assert eta1.shape == eta0.shape
    assert log_det.shape == (pf.num_particles,)
    assert_all_finite(eta1)


def test_pf_weight_update(pf, model, z_obs, p_pred):
    """Test importance weight update for invertible particle flow"""
    particles_prev = pf.particles
    eta_0 = model.propagate_particles(particles_prev)
    eta_1, log_det = pf.run_flow(eta_0, z_obs, p_pred)

    weights = pf._update_weights(z_obs, particles_prev, eta_0, eta_1, log_det)
    assert weights.shape == (pf.num_particles,)
    assert tf.reduce_sum(weights).numpy() == pytest.approx(1.0)


def test_pf_run_step(pf, z_obs, p_pred, cfg):
    """Test single PF-PF filtering step returns valid state estimate"""
    x_est = pf.run_step(z_obs, p_pred)
    assert x_est.shape == (cfg.state_dim,)
    assert_all_finite(x_est)

# End-to-End Pipeline Test
def test_full_filtering_pipeline(cfg, model):
    """Test complete EKF + PF-PF tracking pipeline for both modes"""
    # Short sequence for fast testing
    cfg.T = 3
    cfg.n_particles = 30

    # Generate test data
    x_true, z_meas = [], []
    x_curr = cfg.x0_mean
    for _ in range(cfg.T):
        x_curr = model.transition(x_curr, noise=True)
        x_curr = model.reflect_boundaries(x_curr)
        z_meas.append(model.sample_measurement(x_curr).numpy())
        x_true.append(x_curr.numpy())

    # Test both modes
    for mode in ["LEDH", "EDH"]:
        ekf = EKF_Predictor(cfg, model)
        pf = AcousticPFPF(cfg, model, mode=mode)
        est_traj = []

        for t in range(cfg.T):
            P_pred = ekf.predict()
            x_est = pf.run_step(tf.constant(z_meas[t]), P_pred)
            ekf.update(tf.constant(z_meas[t]))
            est_traj.append(x_est.numpy())

        # Validate results
        est_traj = np.array(est_traj)
        assert est_traj.shape == (cfg.T, cfg.state_dim)
        assert_all_finite(est_traj)