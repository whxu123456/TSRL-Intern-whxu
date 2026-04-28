import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.nonlinear_ssm import NonlinearSSMConfig, NonlinearSSMModel
from src.filters.EKF_nonlinear_ssm import EKF_NonlinearSSM
from src.filters.nonlinear_pfpf import NonlinearPFPF
from src.filters.differentiable_pfpf import DifferentiableNonlinearPFPF
from main_nonlinear_ssm_dpf import (
    make_cfg_from_theta,
    theta_to_z,
    z_to_theta,
    compute_log_prior_and_jac,
    log_prior_z,
    pmmh_loglik,
    log_prior,
    run_pmmh,
    run_hmc,
    compute_metrics,
    run_diagnostics
)

tfd = tfp.distributions
tf.random.set_seed(42)
np.random.seed(42)


# Unit Tests for Utility Functions
class TestUtilityFunctions:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.base_cfg = NonlinearSSMConfig(T=5)
        self.base_cfg.n_particles = 5
        self.true_theta = np.array([0.5, 25.0, 8.0, 1.2, 10.0, 1.0], dtype=np.float32)

    def test_make_cfg_from_theta(self):
        """Test configuration generation from parameter vector theta"""
        cfg = make_cfg_from_theta(self.true_theta, self.base_cfg)
        assert cfg.alpha == 0.5
        assert cfg.beta == 25.0
        assert cfg.sigma_v2 == 10.0
        assert cfg.n_particles == 5

    def test_theta_z_conversion(self):
        """Test forward and inverse transformation between theta and z"""
        z = theta_to_z(self.true_theta)
        theta_recover = z_to_theta(z)
        np.testing.assert_allclose(self.true_theta, theta_recover, atol=1e-5)

    def test_compute_log_prior_and_jac(self):
        """Test log prior and Jacobian computation in transformed space"""
        z = theta_to_z(self.true_theta)
        z_tf = tf.convert_to_tensor(z, dtype=tf.float32)
        theta, log_p, log_jac = compute_log_prior_and_jac(z_tf, self.base_cfg)

        assert theta.shape == (6,)
        assert tf.rank(log_p) == 0
        assert tf.rank(log_jac) == 0
        assert np.isfinite(log_p.numpy())
        assert np.isfinite(log_jac.numpy())

    def test_log_prior_z(self):
        """Test log prior evaluation in transformed z space"""
        z = theta_to_z(self.true_theta)
        lp = log_prior_z(z, self.base_cfg)
        assert np.isfinite(lp)

        invalid_z = z.copy()
        invalid_z[4] = 100
        assert log_prior_z(invalid_z, self.base_cfg) == -np.inf

    def test_log_prior(self):
        """Test log prior evaluation in original parameter space"""
        lp = log_prior(self.true_theta, self.base_cfg)
        assert np.isfinite(lp.numpy())


# Unit Tests for Likelihood
class TestLogLikelihood:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = NonlinearSSMConfig(T=5)
        self.cfg.n_particles = 5
        self.model = NonlinearSSMModel(self.cfg)
        self.x_true, self.y_obs = self.model.generate_true_trajectory()
        self.true_theta = np.array([0.5, 25.0, 8.0, 1.2, 10.0, 1.0], dtype=np.float32)

    def test_pmmh_loglik(self):
        """Test log likelihood computation using standard PF-PF for PMMH"""
        ll = pmmh_loglik(self.true_theta, self.cfg, self.y_obs)
        assert np.isfinite(ll)

        invalid_theta = self.true_theta.copy()
        invalid_theta[4] = -1.0
        assert pmmh_loglik(invalid_theta, self.cfg, self.y_obs) == -np.inf


# Unit Tests for MCMC Samplers
class TestMCMC:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = NonlinearSSMConfig(T=5)
        self.cfg.n_particles = 5
        self.model = NonlinearSSMModel(self.cfg)
        self.x_true, self.y_obs = self.model.generate_true_trajectory()

    def test_run_pmmh(self):
        """Test PMMH sampler execution with small sample size"""
        samples, acc_rate, runtime = run_pmmh(
            self.y_obs, self.cfg, self.model, n_samples=10, burnin=2
        )
        assert samples.shape == (8, 6)
        assert 0 <= acc_rate <= 1
        assert runtime > 0

    def test_run_hmc(self):
        """Test HMC sampler execution with differentiable PF-PF"""
        samples, acc_rate, runtime = run_hmc(
            self.y_obs, self.cfg, self.model, n_samples=10, burnin=2
        )
        assert samples.shape == (10, 6)
        assert 0 <= acc_rate <= 1
        assert runtime > 0


# Unit Tests for Metrics
class TestMetrics:
    def test_compute_metrics(self):
        """Test computation of ESS, RMSE, bias, and efficiency metrics"""
        true_params = np.array([0.5, 25.0, 8.0, 1.2, 10.0, 1.0])
        samples = np.random.normal(true_params, 0.1, size=(50, 6))
        metrics = compute_metrics(samples, true_params, runtime=2.0)

        assert np.isfinite(metrics["mean_ess"])
        assert np.isfinite(metrics["ess_per_second"])
        assert np.all(np.isfinite(metrics["bias"]))
        assert np.all(np.isfinite(metrics["rmse"]))
        assert metrics["runtime"] == 2.0


# Unit Tests for Filters
class TestFilters:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.cfg = NonlinearSSMConfig(T=10)
        self.cfg.n_particles = 20
        self.model = NonlinearSSMModel(self.cfg)
        self.ekf = EKF_NonlinearSSM(self.cfg, self.model)
        self.x_true, self.y_obs = self.model.generate_true_trajectory()

    def test_nonlinear_pfpf(self):
        """Test standard PF-PF filtering on nonlinear SSM"""
        pf = NonlinearPFPF(self.cfg, self.model, self.ekf, flow_method="LEDH")
        x_est, ess, runtime = pf.run_filter(self.y_obs)
        assert x_est.shape == (self.cfg.T, 1)
        assert ess.shape == (self.cfg.T,)
        assert np.mean(ess) > 0

    def test_differentiable_pfpf(self):
        """Test differentiable PF-PF and log-likelihood computation"""
        dpf = DifferentiableNonlinearPFPF(
            self.cfg, self.model, self.ekf, flow_method="LEDH", epsilon=0.5
        )
        loglik = dpf.compute_differentiable_log_likelihood(
            self.y_obs, tf.convert_to_tensor([0.5, 25.0, 8.0, 1.2, 10.0, 1.0])
        )
        assert np.isfinite(loglik.numpy())


# Full Integration Test
def test_full_pipeline():
    """Full end-to-end integration test: data generation, filtering, MCMC, metrics, diagnostics"""
    print("\n" + "=" * 80)
    print("Running Full Differentiable PF-PF Pipeline Test")
    print("=" * 80)

    cfg = NonlinearSSMConfig(T=5)
    cfg.n_particles = 5
    model = NonlinearSSMModel(cfg)
    x_true, y_obs = model.generate_true_trajectory()
    true_params = np.array([cfg.alpha, cfg.beta, cfg.gamma, cfg.delta, cfg.sigma_v2, cfg.sigma_w2], dtype=np.float32)

    pmmh_samples, pmmh_acc, pmmh_runtime = run_pmmh(y_obs, cfg, model, n_samples=10, burnin=2)
    assert pmmh_samples.shape[0] == 8
    assert pmmh_acc >= 0

    hmc_samples, hmc_acc, hmc_runtime = run_hmc(y_obs, cfg, model, n_samples=10, burnin=2)
    assert hmc_samples.shape[0] == 10
    assert hmc_acc >= 0

    pmmh_metrics = compute_metrics(pmmh_samples, true_params, pmmh_runtime)
    hmc_metrics = compute_metrics(hmc_samples, true_params, hmc_runtime)
    assert np.isfinite(pmmh_metrics["mean_ess"])
    assert np.isfinite(hmc_metrics["mean_ess"])

    run_diagnostics(y_obs, cfg, true_params)

    print("Full pipeline test completed successfully")


if __name__ == "__main__":
    test_full_pipeline()