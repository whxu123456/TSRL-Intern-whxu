import numpy as np
import tensorflow_probability as tfp
import enum
import pykalman

tfd = tfp.distributions

'''
Adapted from scripts.simple_linear_common.py
Add the soft-resampling method
remove the kf_loglikelihood function because pykalman fix the bug
'''

class ResamplingMethodsEnum(enum.IntEnum):
    MULTINOMIAL = 0
    SYSTEMATIC = 1
    STRATIFIED = 2
    REGULARIZED = 3
    VARIANCE_CORRECTED = 4
    OPTIMIZED = 5
    KALMAN = 6
    CORRECTED = 7
    SOFT = 8

def get_data(transition_matrix, observation_matrix, transition_covariance, observation_covariance, T=100,
             random_state=None):
    '''
    Generate the observation data as the baseline.
    '''
    if random_state is None:
        random_state = np.random.RandomState()
    kf = pykalman.KalmanFilter(transition_matrix, observation_matrix, transition_covariance, observation_covariance)
    sample = kf.sample(T, random_state=random_state)
    return sample[1].data.astype(np.float32), kf