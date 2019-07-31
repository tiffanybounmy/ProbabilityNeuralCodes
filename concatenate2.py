#!/usr/bin/env python3

"""
@authors: Tiffany Bounmy
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
import matplotlib
import matplotlib.pyplot as plt
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
from neural_proba import distrib
from neural_proba import tuning_curve
from neural_proba import voxel
from neural_proba import experiment
from neural_proba import fmri

import utils

## Concatenate to get r2 scores for all subjects
# The parameters related to the scheme
scheme_array = ['gaussian_ppc', 'sigmoid_ppc', 'gaussian_dpc', 'sigmoid_dpc', 'rate']
n_schemes = len(scheme_array)

# The parameters related to the tuning curves to be explored
N_array = np.array([2, 3, 4, 5, 6, 7, 8, 10, 14, 20])

# The number of N to be tested
n_N = len(N_array)

# The number of fractions tested "(related to W)
n_fractions = 20

# Sparsity exponents
sparsity_exp_array = np.array([1, 2, 4, 8])
n_sparsity_exp = len(sparsity_exp_array)

# The number of subjects
n_subjects = 1000

# The number of directions
n_directions = 16

# Directions
directions = np.array(list(itertools.product([0,1], repeat=4)))

# The number of sessions
n_sessions = 4

# The number of stimuli per session
n_stimuli = 380

# Way to compute the distributions from the sequence
distrib_type = 'transition' # or 'bernoulli'

# SNR as defined by ||signal||²/(||signal||²+||noise||²)
snr = 0.1

# Initialialise
r2_raw_test_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))
r2_raw_train_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))
rho_raw_test_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))
rho_raw_train_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))

r2_true_test_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))
r2_true_train_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))
rho_true_test_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))
rho_true_train_all = np.zeros((n_schemes, n_schemes, n_fractions, n_subjects, n_sessions))

def concatenate(k_subject):
	r2_raw_train = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/r2_raw_train_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')
	rho_raw_train = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/rho_raw_train_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')
	r2_true_train = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/r2_true_train_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')
	rho_true_train = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/rho_true_train_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')

	r2_raw_test = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/r2_raw_test_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')
	rho_raw_test = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/rho_raw_test_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')
	r2_true_test = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/r2_true_test_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')
	rho_true_test = np.load('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/rho_true_test_snr'+str(snr)+'_subj'+str(k_subject)+'.npy')
	
	r2_raw_train_all[:, :, :, k_subject, :] = r2_raw_train
	rho_raw_train_all[:, :, :, k_subject, :] = rho_raw_train
	r2_true_train_all[:, :, :, k_subject, :] = r2_true_train
	rho_true_train_all[:, :, :, k_subject, :] = rho_true_train

	r2_raw_test_all[:, :, :, k_subject, :] = r2_raw_test
	rho_raw_test_all[:, :, :, k_subject, :] = rho_raw_test
	r2_true_test_all[:, :, :,  k_subject, :] = r2_true_test
	rho_true_test_all[:, :, :, k_subject, :] = rho_true_test


for k_subject in range(n_subjects):
	concatenate(k_subject)

# Save the final scores 
np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_r2_raw_train_snr'+str(snr)+'.npy', r2_raw_train_all)
np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_rho_raw_train_snr'+str(snr)+'.npy', rho_raw_train_all)
np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_r2_true_train_snr'+str(snr)+'.npy', r2_true_train_all)
np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_rho_true_train_snr'+str(snr)+'.npy', rho_true_train_all)

np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_r2_raw_test_snr'+str(snr)+'.npy', r2_raw_test_all)
np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_rho_raw_test_snr'+str(snr)+'.npy', rho_raw_test_all)
np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_r2_true_test_snr'+str(snr)+'.npy', r2_true_test_all)
np.save('/volatile/bounmy/output/results2/snr0.1/'+str(distrib_type)+'2/all/'+str(n_subjects)+'subjects_rho_true_test_snr'+str(snr)+'.npy', rho_true_test_all)
