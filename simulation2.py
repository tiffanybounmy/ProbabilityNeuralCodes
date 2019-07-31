#!/usr/bin/env python3

"""
@author: Sébastien Demortain, Tiffany Bounmy
"""

#%%
# Import useful modules
import os
import scipy
import random as rand
from scipy import io as sio
from scipy import stats
from scipy.stats.stats import pearsonr
import numpy as np
width = 18
height = 16
import pandas as pd

import pickle
import itertools
import time

import copy

import multiprocessing as mp

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import neural_proba
from neural_proba import import_distrib_param
from neural_proba import distrib
from neural_proba import tuning_curve
from neural_proba import voxel
from neural_proba import experiment
from neural_proba import fmri

import utils

#%%
## Properties of the models

#Here are the properties related to :
#- the tuning curves (types, number, variance, ...) both true and fitted. 
#- the neural mixture (sparsities)
#- the subjects
#- the sessions
#- the SNR
#- the type of linear regression performed

# All parameters are here

# Define the seed to reproduce results from random processes
rand.seed(4)

# INPUTS

# The parameters related to the scheme
scheme_array = ['gaussian_ppc', 'sigmoid_ppc', 'gaussian_dpc', 'sigmoid_dpc', 'rate']
n_schemes = len(scheme_array)

N_array = np.array([2, 3, 4, 5, 6, 7, 8, 10, 14, 20])

t_mu_gaussian_array = np.array([0.15, 0.12, 0.1, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2])
t_conf_gaussian_array = np.array([0.25, 0.2, 0.15, 0.12, 0.10, 9e-2, 8e-2, 6e-2, 4e-2, 3e-2])

t_mu_sigmoid_array = np.sqrt(2*np.pi)/4*t_mu_gaussian_array
t_conf_sigmoid_array = np.sqrt(2*np.pi)/4*t_conf_gaussian_array

optimal_k_fit_N_array = np.array([4, 0, 3, 0]).astype(int)


optimal_fit_N_array = np.zeros_like(optimal_k_fit_N_array).astype(float)
optimal_t_mu_array = np.zeros_like(optimal_k_fit_N_array).astype(float)
optimal_t_conf_array = np.zeros_like(optimal_k_fit_N_array).astype(float)

for k_fit_scheme in range(n_schemes - 1):    # We exclude rate coding
    optimal_fit_N_array[k_fit_scheme] = N_array[optimal_k_fit_N_array[k_fit_scheme]]    # Optimal N
    # Now we fill values for optimal mu and t
    if k_fit_scheme % 2 == 0:    # Gaussian case
        t_mu_tmp = t_mu_gaussian_array[optimal_k_fit_N_array[k_fit_scheme]]
        t_conf_tmp = t_conf_gaussian_array[optimal_k_fit_N_array[k_fit_scheme]]
        optimal_t_mu_array[k_fit_scheme] = t_mu_tmp
        optimal_t_conf_array[k_fit_scheme] = t_conf_tmp
    else:
        optimal_t_mu_array[k_fit_scheme] = t_mu_sigmoid_array[optimal_k_fit_N_array[k_fit_scheme]]
        optimal_t_conf_array[k_fit_scheme] = t_conf_sigmoid_array[optimal_k_fit_N_array[k_fit_scheme]]

# Lower and upper bounds of the encoded summary quantity (for tuning curves)
tc_lower_bound_mu = 0
tc_upper_bound_mu = 1
tc_lower_bound_conf = 1.1
# we define the upper bound to be a bit away from the highest uncertainty
tc_upper_bound_conf = 2.6

# Rate coding scaling parameters between probability and confidence neurons
mu_sd = 0.219    # Std of the signal of mu's
conf_sd = 0.284    # Std of the signal of conf's

# The number of possible N
n_N = len(N_array)

# The number of fractions tested (related to W)
n_fractions = 20

# Sparsity exponents
sparsity_exp_array = np.array([1, 2, 4, 8])
n_sparsity_exp = len(sparsity_exp_array)
between_population_sparsity_array = np.array([[0.5, 0.5], [0.25, 0.75], [0, 1], [0.75, 0.25], [1, 0]])

# The number of subjects
n_subjects = 1000

# Create the matrix with all the possibilities (mu or 1-mu encoded in each session) of size 16x4
directions = np.array(list(itertools.product([0,1], repeat=4)))
n_directions = len(directions)

# The number of sessions
n_sessions = 4

# Experimental options
n_stimuli = 380    # The number of stimuli per session
between_stimuli_duration = 1.3
min_break_time = 8
max_break_time = 12
min_n_local_regular_stimuli = 12
max_n_local_regular_stimuli = 18

# Transition proba or Bernoulli proba
distrib_type = 'transition'

# Load the corresponding data
[p1_dist_array, p1_mu_array, p1_sd_array] = neural_proba.import_distrib_param(n_subjects, n_sessions, n_stimuli,
                                                                                      distrib_type)

# SNR as defined by ||signal||²/(||signal||²+||noise||²)
snr = 0.1

# fMRI info
final_frame_offset = 10  # Frame recording duration after the last stimulus has been shown
initial_frame_time = 0
dt = 0.125  # Temporal resolution of the fMRI scanner

between_scans_duration = 2  # in seconds
final_scan_offset = 10  # Scan recording duration after the last stimulus has been shown

# Type of regression
regr = linear_model.LinearRegression(fit_intercept=True, n_jobs=-1)
regr2 = linear_model.LinearRegression(fit_intercept=True, n_jobs=-1) 


#%%
def create_design_matrix(k_subject):
    '''Creates the design matrix for each subject from the ideal observer model output'''
    
    # Initialization of the design matrices and their zscore versions
    X = [[[None for k_session in range(n_sessions)] for k_direction in range(n_directions)]
         for k_fit_scheme in range(n_schemes)]
    
    ### WE BEGIN BY CREATING THE DESIGN MATRIX X
    start = time.time()
    
    # Experimental design information
    eps = 1e-5  # For floating points issues
    initial_time = between_stimuli_duration + eps
    final_time_tmp = between_stimuli_duration * (n_stimuli + 1) + eps
    # Every 15+-3 trials : one interruption of 8-12s
    stimulus_onsets = np.linspace(initial_time, final_time_tmp, n_stimuli)
    # We add some time to simulate breaks
    stimulus = 0

    while True:
        # Number of regularly spaced stimuli
        n_local_regular_stimuli = rand.randint(min_n_local_regular_stimuli, max_n_local_regular_stimuli)
        stimulus_shifted = stimulus + n_local_regular_stimuli  # Current stimulus before the break
        if stimulus_shifted > n_stimuli:  # The next break is supposed to occur after all stimuli are shown
            break
        stimulus_onsets[stimulus_shifted:] += rand.randint(min_break_time,
                                                           max_break_time) - between_stimuli_duration  # We consider a break of 8-12s
        stimulus = stimulus_shifted

    stimulus_durations = dt * np.ones_like(stimulus_onsets)  # Dirac-like stimuli

    # fMRI information
    final_time = stimulus_onsets[-1]
    final_frame_time = final_time + final_frame_offset

    initial_scan_time = initial_frame_time + between_scans_duration
    final_scan_time = final_time + final_scan_offset
    scan_times = np.arange(initial_scan_time, final_scan_time, between_scans_duration)
    
    ### Loop over the directions:
    for k_direction in range(n_directions):
        ### Loop over the sessions : we start with it in order to have the same length whatever N_fit is
        for k_session in range(n_sessions):
            # Get the data of interest
            if directions[k_direction, k_session] == 0:
                mu = p1_mu_array[k_subject, k_session, :n_stimuli]
                dist = p1_dist_array[k_subject, k_session, :, :n_stimuli]
            else:
                mu = 1 - p1_mu_array[k_subject, k_session, :n_stimuli]
                dist = np.flipud(p1_dist_array[k_subject, k_session, :, :n_stimuli])
            sigma = p1_sd_array[k_subject, k_session, :n_stimuli]
            conf = -np.log(sigma)

            # Formatting
            simulated_distrib = [None for k in range(n_stimuli)]
            for k in range(n_stimuli):
                # Normalization of the distribution
                norm_dist = dist[:, k] * (len(dist[1:, k]) - 1) / np.sum(dist[1:, k])
                simulated_distrib[k] = distrib(mu[k], sigma[k], norm_dist)

            # Creation of fmri object
            simu_fmri = fmri(initial_frame_time, final_frame_time, dt, scan_times)

            # Creation of experiment object
            exp = experiment(initial_time, final_time, n_sessions, stimulus_onsets, stimulus_durations, simulated_distrib)

            ### LOOP OVER THE SCHEME
            for k_fit_scheme in range(n_schemes):

                # Current schemes
                fit_scheme = scheme_array[k_fit_scheme]

                # We replace the right value of the "t"'s according to the type of tuning curve and the N
                if fit_scheme.find('gaussian') != -1:
                    fit_N = optimal_fit_N_array[k_fit_scheme]
                    fit_t_mu = optimal_t_mu_array[k_fit_scheme]
                    fit_t_conf = optimal_t_conf_array[k_fit_scheme]

                    fit_tc_type = 'gaussian'
                    # Creation of the true tuning curve objects
                    fit_tc_mu = tuning_curve(fit_tc_type, fit_N, fit_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
                    fit_tc_conf = tuning_curve(fit_tc_type, fit_N, fit_t_conf, tc_lower_bound_conf,
                                                 tc_upper_bound_conf)

                elif fit_scheme.find('sigmoid') != -1:
                    fit_N = optimal_fit_N_array[k_fit_scheme]
                    fit_t_mu = optimal_t_mu_array[k_fit_scheme]
                    fit_t_conf = optimal_t_conf_array[k_fit_scheme]

                    fit_tc_type = 'sigmoid'
                    # Creation of the true tuning curve objects
                    fit_tc_mu = tuning_curve(fit_tc_type, fit_N, fit_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
                    fit_tc_conf = tuning_curve(fit_tc_type, fit_N, fit_t_conf, tc_lower_bound_conf,
                                                 tc_upper_bound_conf)

                if fit_scheme.find('ppc') != -1:
                    fit_tc = [fit_tc_mu, fit_tc_conf]
                elif fit_scheme.find('dpc') != -1:
                    fit_tc = [fit_tc_mu]
                elif fit_scheme.find('rate') != -1:
                    fit_tc = []

                # Regressor and BOLD computation
                X[k_fit_scheme][k_direction][k_session] = simu_fmri.get_regressor(exp, fit_scheme, fit_tc)

                # Rescale the regressors for rate code
                if fit_scheme.find('rate') != -1:
                    X[k_fit_scheme][k_direction][k_session][:, 1] = mu_sd/conf_sd * X[k_fit_scheme][k_direction][k_session][:, 1]            

    end = time.time()
    print('Design matrix creation : Subject n'+str(k_subject)+' is done ! Time elapsed : '+str(end-start)+'s')
    return X
#%%

def whiten_design_matrix(k_subject):
    '''Applies whitening to the created design matrix'''
    X = create_design_matrix(k_subject)
    whitening_done = False
    whitening_start = time.time()
    # Whiten the design matrices

    # Whitening matrix
    white_mat = sio.loadmat('data/simu/whitening_matrix.mat')
    W = white_mat['W']
    # Complete the in-between session "holes"
    W[300:600, 300:600] = W[20:320, 20:320]
    
    if not whitening_done:
        # Multiplying the zscored X with the whitening matrix
        for k_fit_scheme, k_direction, k_session in itertools.product(range(n_schemes), range(n_directions), range(n_sessions)):
            X_tmp = copy.deepcopy(X[k_fit_scheme][k_direction][k_session])    # Just to lighten code
            rows_dim, columns_dim = X_tmp.shape
            X_tmp = np.matmul(W[:rows_dim, :rows_dim], X_tmp)
            X[k_fit_scheme][k_direction][k_session] = copy.deepcopy(X_tmp)
            #del X_tmp
    
    whitening_done = True
    
    X_after_whitening = copy.deepcopy(X)
    whitening_end = time.time()
    print('Matrix whitening done in '+str(whitening_end-whitening_start)+' seconds')
    return X_after_whitening
#%%
def compute_response_vector_weights(k_subject):
    '''Computes y from X and weights'''
    X = whiten_design_matrix(k_subject)
    
    # Creation of y from X to save computational resources
    # Initialization of the response vectors
    y = [[[[None for k_session in range(n_sessions)] for k_direction in range(n_directions)] for k_fraction in range(n_fractions)]
    for k_true_scheme in range(n_schemes)]
    
    # Initialization of the weights
    weights = [[None for k_fraction in range(n_fractions)] 
                for k_true_scheme in range(n_schemes)]

    ### LOOP OVER THE SCHEME
    for k_true_scheme in range(n_schemes):
        true_scheme = scheme_array[k_true_scheme]
        # We replace the right value of the "t"'s according to the type of tuning curve
    
        if true_scheme.find('gaussian') != -1:
            true_N = int(optimal_fit_N_array[k_true_scheme])
            true_t_mu = optimal_t_mu_array[k_true_scheme]
            true_t_conf = optimal_t_conf_array[k_true_scheme]
            true_tc_type = 'gaussian'
            # Creation of the true tuning curve objects
            true_tc_mu = tuning_curve(true_tc_type, true_N, true_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
            true_tc_conf = tuning_curve(true_tc_type, true_N, true_t_conf, tc_lower_bound_conf,
                                         tc_upper_bound_conf)
    
    
        elif true_scheme.find('sigmoid') != -1:
            true_N = int(optimal_fit_N_array[k_true_scheme])
            true_t_mu = optimal_t_mu_array[k_true_scheme]
            true_t_conf = optimal_t_conf_array[k_true_scheme]
            true_tc_type = 'sigmoid'
            # Creation of the true tuning curve objects
            true_tc_mu = tuning_curve(true_tc_type, true_N, true_t_mu, tc_lower_bound_mu, tc_upper_bound_mu)
            true_tc_conf = tuning_curve(true_tc_type, true_N, true_t_conf, tc_lower_bound_conf,
                                         tc_upper_bound_conf)
    
    
        # We consider combinations of population fractions for PPC and rate codes
        if true_scheme.find('ppc') != -1 or true_scheme.find('rate') != -1:
            # The number of population fraction tested (related to W)
            population_fraction_array = copy.deepcopy(between_population_sparsity_array)
        elif true_scheme.find('dpc') != -1:  # DPC case
            population_fraction_array = copy.deepcopy(np.array([[1]]))
        n_population_fractions = len(population_fraction_array)
    
        if true_scheme.find('ppc') != -1:
            true_tc = [true_tc_mu, true_tc_conf]
        elif true_scheme.find('dpc') != -1:
            true_tc = [true_tc_mu]
        elif true_scheme.find('rate') != -1:
            true_tc = []
    
        ### LOOP OVER THE SUBJECTS
        for k_subject in range(n_subjects):
    
            ### LOOP OVER THE W's
            # The number of subpopulation fractions acc. to the scheme
            n_subpopulation_fractions = int(n_fractions / n_population_fractions)
            fraction_counter = 0
            for k_subpopulation_fraction in range(n_subpopulation_fractions):
                for k_population_fraction, population_fraction in enumerate(population_fraction_array):
                    # The number of populations acc. to the scheme (2 for PPC and rate, 1 for DPC)
                    n_population = len(population_fraction)
                    if true_scheme.find('ppc') != -1 or true_scheme.find('dpc') != -1:
                        # We consider one sparsity per remainder value of the counter divided by the number
                        # of combinations to be tested
                        
                        subpopulation_sparsity_exp = sparsity_exp_array[fraction_counter % n_sparsity_exp]
                        # Fraction of each neural subpopulation
                        subpopulation_fraction = neural_proba.get_subpopulation_fraction(n_population, true_N,
                                                                                         subpopulation_sparsity_exp)
                    elif true_scheme.find('rate') != -1:  # Rate case
                        subpopulation_fraction = np.array([[1.0],[1.0]])
                        
    
                    # Generate the data from the voxel
                    true_voxel = voxel(true_scheme, population_fraction, subpopulation_fraction, true_tc)
                    n_true_features = n_population * len(subpopulation_fraction[0])
                    weights_tmp = np.reshape(true_voxel.weights, (n_true_features,))
    
                    # Allocation of the weight tensor
                    weights[k_true_scheme][fraction_counter] \
                        = copy.deepcopy(weights_tmp)
                    
                    ### LOOP OVER THE DIRECTIONS 
                    for k_direction in range(n_directions):       
                        ### LOOP OVER THE SESSIONS : simulating the response    
                        for k_session in range(n_sessions):
                            # We use X to compute y in order to save some computation time
                            # Temporary variables to lighten the reading
                            y[k_true_scheme][fraction_counter][k_direction][k_session] = copy.deepcopy(np.matmul(X[k_true_scheme][k_direction][k_session], weights_tmp))
                            
                    fraction_counter += 1
    
    # y_without_noise = copy.deepcopy(y)

    # Normalization for each scheme

    y_sd_all = np.zeros((n_schemes, n_fractions, n_directions, n_sessions))
    
    for k_true_scheme, k_fraction, k_direction, k_session in itertools.product(range(n_schemes), range(n_fractions), range(n_directions), range(n_sessions)):
        y_sd_all[k_true_scheme, k_fraction, k_direction, k_session] = np.std(y[k_true_scheme][k_fraction][k_direction][
                                k_session])
            
    y_sd = np.zeros(n_schemes)
    
    for k_true_scheme in range(n_schemes):
        y_sd[k_true_scheme] = np.mean(y_sd_all[k_true_scheme, :, :, :])
        for k_fraction, k_direction, k_session in itertools.product(range(n_fractions), range(n_directions), range(n_sessions)):
            y[k_true_scheme][k_fraction][k_direction][k_session] = copy.deepcopy(y[k_true_scheme][k_fraction][k_direction][
                                k_session]/y_sd[k_true_scheme])
        
        for k_fraction in range(n_fractions):
            weights[k_true_scheme][k_fraction]= copy.deepcopy(weights[k_true_scheme][k_fraction]/y_sd[k_true_scheme])
    
    y_without_noise = copy.deepcopy(y)

    # Compute the amplitude of the noise
    noise_sd = np.zeros(n_schemes)
    for k_true_scheme in range(n_schemes):
        noise_sd[k_true_scheme] = np.sqrt(1 / snr - 1)  # std of the added gaussian noise
    print(noise_sd)
    
    # Add the noise
    for k_true_scheme, k_fraction, k_direction, k_session in itertools.product(range(n_schemes), range(n_fractions), range(n_directions), range(n_sessions)):
        y[k_true_scheme][k_fraction][k_direction][k_session] = y[k_true_scheme][k_fraction][k_direction][k_session] + np.random.normal(0, noise_sd[k_true_scheme], len(y[k_true_scheme][k_fraction][k_direction][k_session]))
            
    # y_with_noise = copy.deepcopy(y)

    # Create the filtering design matrices and filters out the response
    
    for k_true_scheme, k_fraction, k_direction, k_session in itertools.product(range(n_schemes), range(n_fractions), range(n_directions), range(n_sessions)):    
        y_tmp = copy.deepcopy(y[k_true_scheme][k_fraction][k_direction][k_session])
        N = len(y_tmp)    # Resolution of the signal
        K = 11    # Highest order of the filter
        n_grid = np.linspace(0, N-1, N, endpoint=True)    # 1D grid over values
        k_grid = np.linspace(2, K, K-1, endpoint=True)    # 1D grid over orders
        X_filter = np.zeros((N, K-1))
        for kk, k in enumerate(k_grid):
            X_filter[:, kk] = np.sqrt(2/N) * np.cos(np.pi*(2*n_grid+1)/(2*N)*(k-1))
        y_tmp = copy.deepcopy(y_tmp - np.matmul(np.matmul(X_filter, np.transpose(X_filter)), y_tmp))    # Regression
        y[k_true_scheme][k_fraction][k_direction][k_session] = copy.deepcopy(y_tmp)
    
    # y_after_filtering = copy.deepcopy(y)
    print('Response vectors and weights computed!')
    return X, y_without_noise, y, weights
#%%
def z_scoring(k_subject):
    '''Z-score of X and y and readjust the weights after z-scoring'''
    
    X, y_without_noise, y, weights = compute_response_vector_weights(k_subject)
    
    # Z-scoring of X and y
    
    scaling_factor_X = 0.01
    snr_factor = 1 #np.sqrt(1/snr-1)
    
    # Initialization
    Xz = [[[None for k_session in range(n_sessions)] for k_direction in range(n_directions)]
         for k_fit_scheme in range(n_schemes)]
    
    yz = [[[[None for k_session in range(n_sessions)] for k_direction in range(n_directions)
            ] for k_fraction in range(n_fractions)] for k_true_scheme in range(n_schemes)]
          
    yz_without_noise = [[[[None for k_session in range(n_sessions)] for k_direction in range(n_directions)] for k_fraction in range(n_fractions)]
                        for k_true_scheme in range(n_schemes)]
    
    for k_fit_scheme, k_direction, k_session in itertools.product(range(n_schemes), range(n_directions), range(n_sessions)):
        Xz[k_fit_scheme][k_direction][k_session] = np.zeros_like(X[k_fit_scheme][k_direction][k_session])
    
    # Manual Z-scoring of regressors inside the session
    for k_fit_scheme, k_direction, k_session in itertools.product(range(n_schemes), range(n_directions), range(n_sessions)):
        n_fit_features = len(X[k_fit_scheme][k_direction][0][0])
        for feature in range(n_fit_features):
            X_mean = np.mean(X[k_fit_scheme][k_direction][k_session][:, feature], axis=0)
            Xz[k_fit_scheme][k_direction][k_session][:, feature]\
                = (copy.deepcopy(X[k_fit_scheme][k_direction][k_session][:, feature]) - X_mean * np.ones_like(X[k_fit_scheme][k_direction][k_session][:, feature])) / scaling_factor_X  # Centering + Scaling
        # End of z-scoring
            
    for k_true_scheme in range(n_schemes):
        for k_fraction, k_direction in itertools.product(range(n_fractions), range(n_directions)):    
            # Z-scoring of y
            for k_session in range(n_sessions):
                y_mean = np.mean(y[k_true_scheme][k_fraction][k_direction][k_session], axis=0)
    
                yz[k_true_scheme][k_fraction][k_direction][k_session] = \
                    (copy.deepcopy(y[k_true_scheme][k_fraction][k_direction][k_session] - y_mean))    # Centering+standardization
                yz_without_noise[k_true_scheme][k_fraction][k_direction][k_session] = \
                    (copy.deepcopy(y_without_noise[k_true_scheme][k_fraction][k_direction][k_session] - y_mean))    # Centering+standardization
    
            ### End of z-scoring of y
        
    # Reajusting the weights after zscoring
    for k_true_scheme, k_fraction in itertools.product(range(n_schemes), range(n_fractions)):    
        for feature in range(weights[k_true_scheme][k_fraction].shape[0]):
            weights[k_true_scheme][k_fraction][feature] = weights[k_true_scheme][k_fraction][feature]*scaling_factor_X/snr_factor
            
    return Xz, yz, yz_without_noise, weights
#%%
def cross_validation(k_subject):
    Xz, yz, yz_without_noise, weights = z_scoring(k_subject)
    
    print('Cross-validation starts...')
    cv_start = time.time()
    # The loops
    # The quantity to be computed during the cross validation
    r2_raw_test = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    r2_raw_train = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    rho_raw_test = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    rho_raw_train = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    
    r2_true_test = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    r2_true_train = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    rho_true_test = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    rho_true_train = np.zeros((n_schemes, n_schemes, n_fractions, n_sessions))
    
    # simulate the direction encoded by the subject
    k_direction = rand.randint(0, n_directions-1) 
    
    ### BEGINNING OF LOOPS OVER HYPERPARAMETERS
    for k_fit_scheme, k_true_scheme in itertools.product(range(n_schemes), range(n_schemes)):
        for k_fraction in range(n_fractions):
            # Current cross-validation matrix and response
            X_cv = copy.deepcopy(Xz[k_fit_scheme])
            y_cv = copy.deepcopy(yz[k_true_scheme][k_fraction][k_direction])
            y_without_noise_cv = copy.deepcopy(yz_without_noise[k_true_scheme][k_fraction][k_direction])
    
            # LOOP OVER SESSIONS (CV)
            for k_session in range(n_sessions):
                # Train the model using the training set
                y_train = copy.deepcopy(np.concatenate(y_cv[:k_session]+y_cv[k_session+1:], axis=0))
                y_without_noise_train = copy.deepcopy(np.concatenate(y_without_noise_cv[:k_session]+y_without_noise_cv[k_session+1:], axis=0))
                
                # Initialise the scores that will be saved for all directions
                r2_raw_train_all = np.zeros(n_directions) # save r2_scores for all directions 
                rho_raw_train_all = np.zeros(n_directions) # save rho scores for all directions
                r2_true_train_all = np.zeros(n_directions)
                rho_true_train_all = np.zeros(n_directions)
                
                weights_train = []
                for k_dir in range(n_directions): 
                    X_train = copy.deepcopy(np.concatenate(X_cv[k_direction][:k_session]+X_cv[k_direction][k_session+1:], axis=0))
                        
                    regr.fit(X_train, y_train)
                    y_hat_train = regr.predict(X_train)
                    weights_train.append(regr.coef_)
                        
                    # Train results 
                    r2_raw_train_all[k_dir] = r2_score(y_train, y_hat_train)
                    rho_raw_train_all[k_dir] = pearsonr(y_train, y_hat_train)[0]
                    r2_true_train_all[k_dir] = r2_score(y_without_noise_train, y_hat_train)
                    rho_true_train_all[k_dir] = pearsonr(y_without_noise_train, y_hat_train)[0]
                    
                # Save results for later        
                best_train_direction_i = np.argwhere(rho_raw_train_all == np.amax(rho_raw_train_all)).flatten().tolist()
                r2_raw_train[k_fit_scheme, k_true_scheme, k_fraction, k_session] = r2_raw_train_all[
                        best_train_direction_i[0]]
                rho_raw_train[k_fit_scheme, k_true_scheme, k_fraction, k_session] = rho_raw_train_all[
                        best_train_direction_i[0]]
                r2_true_train[k_fit_scheme, k_true_scheme, k_fraction, k_session] = r2_true_train_all[
                        best_train_direction_i[0]]
                rho_true_train[k_fit_scheme, k_true_scheme, k_fraction, k_session] = rho_true_train_all[
                        best_train_direction_i[0]]
                
                #regr.coef_ = weights_train[index_max_r2_score_train[0]]
                X_train = copy.deepcopy(np.concatenate(X_cv[best_train_direction_i[0]][:k_session]+X_cv[best_train_direction_i[0]][k_session+1:], axis=0))
                regr.fit(X_train, y_train)
                
                # Make predictions using the testing set
                
                y_test = copy.deepcopy(y_cv[k_session])
                y_without_noise_test = copy.deepcopy(y_without_noise_cv[k_session])
    
                r2_raw_test_all = np.zeros(len(best_train_direction_i))
                rho_raw_test_all = np.zeros(len(best_train_direction_i))
                r2_true_test_all = np.zeros(len(best_train_direction_i))
                rho_true_test_all = np.zeros(len(best_train_direction_i))
    
                for k_dir2 in range(len(best_train_direction_i)): # test with the two different directions
                    X_test = copy.deepcopy(X_cv[best_train_direction_i[k_dir2]][k_session])
                    y_pred = regr.predict(X_test)
                    y_pred_tmp = np.transpose(np.array([y_pred]))
                    
                    # Second fit 
                    regr2.fit(y_pred_tmp, y_test)
                    y_pred2 = regr2.predict(y_pred_tmp)
                    
                    # Test results
                    r2_raw_test_all[k_dir2] = r2_score(y_test, y_pred2)
                    rho_raw_test_all[k_dir2] = pearsonr(y_test, y_pred2)[0]
                    r2_true_test_all[k_dir2] = r2_score(y_without_noise_test, y_pred2)
                    rho_true_test_all[k_dir2] = pearsonr(y_without_noise_test, y_pred2)[0]
    
                best_test_direction = np.argwhere(rho_raw_test_all == np.amax(rho_raw_test_all)).flatten().tolist() # returns the index associated with the greatest r2_score
    
                r2_raw_test[k_fit_scheme, k_true_scheme, k_fraction, k_session] = r2_raw_test_all[
                        best_test_direction[0]]
                r2_true_test[k_fit_scheme, k_true_scheme, k_fraction, k_session] = r2_true_test_all[
                        best_test_direction[0]]
                rho_raw_test[k_fit_scheme, k_true_scheme, k_fraction, k_session] = rho_raw_test_all[
                        best_test_direction[0]]
                rho_true_test[k_fit_scheme, k_true_scheme, k_fraction, k_session] = rho_true_test_all[
                        best_test_direction[0]]
            
        print('Scheme n'+str(k_true_scheme)+' fitted with scheme n'+str(k_fit_scheme)+'.')
    cv_end = time.time()
    print('Cross validation done in '+str(cv_end-cv_start)+' seconds.')
    return r2_raw_train, rho_raw_train, r2_true_train, rho_true_train, r2_raw_test, r2_true_test, rho_raw_test, rho_true_test

def get_scores(k_subject):
    '''Retrieves rho and r2 scores and saves these values'''
    r2_raw_train, rho_raw_train, r2_true_train, rho_true_train, r2_raw_test, r2_true_test, rho_raw_test, rho_true_test = cross_validation(k_subject)
    
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/r2_raw_train_snr'+str(snr)+'_subj'+str(k_subject)+'.npy', r2_raw_train)
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/rho_raw_train_snr'+str(snr)+'_subj' + str(k_subject) + '.npy', rho_raw_train)
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/r2_true_train_snr'+str(snr)+'_subj' + str(k_subject) + '.npy', r2_true_train)
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/rho_true_train_snr'+str(snr)+'_subj' + str(k_subject) + '.npy', rho_true_train)
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/r2_raw_test_snr'+str(snr)+'_subj' + str(k_subject) + '.npy', r2_raw_test)
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/rho_raw_test_snr'+str(snr)+'_subj' + str(k_subject) + '.npy', rho_raw_test)
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/r2_true_test_snr'+str(snr)+'_subj' + str(k_subject) + '.npy', r2_true_test)
    np.save('/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/cross_validation/output/results2/snr0.1/'+str(distrib_type)+'/rho_true_test_snr'+str(snr)+'_subj' + str(k_subject) + '.npy', rho_true_test)

#%% 
for k_subject in range(n_subjects):
    subj_start = time.time()  
    get_scores(k_subject)
    subj_end = time.time()
    print('Subject done after '+str(subj_end-subj_start)+' seconds.')
print('Simulation done!')
