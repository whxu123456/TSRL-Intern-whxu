# TSRL-Intern-whxu
Candidate project for the 2026 Machine Learning Center of Excellence Summer Associate – Time Series &amp; Reinforcement Learning Internship.

# Particle Flow Filter and Differentiable Particle Filter Experiments
This repository contains the Python implementations and experimental scripts for the report **Particle Flow Filter and Differentiable Particle Filter**. The code covers classical filtering methods, Particle Filters, and advanced Particle Flow Filters.

## Project Structure

The experiments are organized by the model type and filtering algorithm used.

### 1. Linear-Gaussian State Space Models (LGSSM)
*   **Goal:** Compare the numerical stability of the Standard Kalman Filter vs. the Joseph Stabilized form.
*   **Key Script:** `main_kalman.py`
*   **Description:** Implements a 4D state / 2D observation linear system. Generates comparison plots for covariance symmetry and condition numbers.

### 2. Stochastic Volatility (SV) Model
*   **Goal:** Evaluate filter performance on a system with multiplicative noise and non-linear observations.
*   **Key Script:** `main_nonlinear.py`
*   **Description:** 
    *   Implements the SV model dynamics.
    *   Compares EKF/UKF (Raw vs Log-Transformed) and Standard Particle Filter.
    *   Generates RMSE tables and Effective Sample Size (ESS) plots.

### 3. Multi-Target Acoustic Tracking (Particle Flow)
*   **Goal:** Replicate the acoustic tracking experiment from Li and Coates (2017) to test Particle Flow Filters.
*   **Key Script:** `main_pfpf_acoustic.py`
*   **Description:** 
    *   Simulates 4 targets moving in a 2D space monitored by 25 sensors.
    *   Implements **PF-PF (LEDH)** and **PF-PF (EDH)**.
    *   Calculates Optimal Mass Transfer (OMAT) error.

### 4. Lorenz 96 (Kernel-Embedded PFF)
*   **Goal:** Test filtering performance on a high-dimensional ($N_x=1000$) chaotic system.
*   **Key Script:** `main_kernel_pff.py`
*   **Description:** 
    *   Implements the Lorenz 96 ODE using RK4 integration.
    *   Compares **Scalar-valued Kernel** vs. **Matrix-valued Kernel** PFF.
    *   Reproduces the particle collapse vs. spread visualization.

### 5. Deterministic vs. Kernel Flows (SV Model)
*   **Goal:** A comparative study of flow-based methods on the Stochastic Volatility model.
*   **Key Script:** `main_dkff_sv.py`
*   **Description:** 
    *   Compares EDH-Log, EDH-Exact, LEDH-Exact, and Kernel PFF.
    *   Analyzes flow magnitude stability.
