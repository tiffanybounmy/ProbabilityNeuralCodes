#! /usr/bin/env python3
# Time-stamp: <>
#%%
import glob
import os
import os.path as op
import sys
import csv
import pandas as pd
import seaborn as sns
import itertools

import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
import numpy as np
from nilearn.input_data import MultiNiftiMasker

from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LinearRegression
from nilearn.image import math_img, mean_img, threshold_img
from nilearn.plotting import plot_glass_brain
from nilearn.image import coord_transform
import pylab as plt
from joblib import Parallel, delayed


#%%
def get_fmri_files_from_subject(rootdir, subject):
    return sorted(glob.glob(os.path.join(rootdir,
                                         "MRI_data/raw_data",
                                         "subj%02d" % subject,
                                         "fMRI/wtraepi*.nii")))

#%%
def get_design_matrices(matrixdir, subject):
    # initialise the dictionary that will contain all design matrices for one sub
    matrices = {'gaussian_ppc_mu':[], 'gaussian_ppc_conf':[],
                'sigmoid_ppc_mu':[], 'sigmoid_ppc_conf':[],
                'gaussian_dpc':[], 'sigmoid_dpc':[],
                'rate_mu':[], 'rate_conf':[]}
    
    # get the path of the design matrices for one sub
    dmtx_path = sorted(glob.glob(op.join(modeldir, 
                                         "subj%02d" % subject,
                                         'dmtx*.pkl')))
    
    # retrieve the design matrices and insert in the directionary wrt the scheme
    for i in range(len(dmtx_path)):
        dmtx = pd.read_pickle(dmtx_path[i])
        for k_scheme in schemes:
            if dmtx_path[i].find(k_scheme) != -1:
                matrices[k_scheme].append(dmtx)

    return matrices

#%%

def compute_global_masker(rootdir):
    mask = op.join(rootdir, 'spm12/tpm/mask_ICV.nii')
    global_mask = math_img('img>0', img= mask)
    masker = MultiNiftiMasker(global_mask, smoothing_fwhm = 1.5,
                              detrend=True, standardize=True)
    masker.fit()
    return masker

#%%
def compute_crossvalidated_r2(fmri_runs, design_matrices, loglabel, logcsvwriter):

    def log(r2_train, r2_test):
        """ log stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, 'training', np.mean(r2_train), np.std(r2_train),
                               np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, 'test', np.mean(r2_test), np.std(r2_test),
                               np.min(r2_test), np.max(r2_test)])
    
    for k_scheme in schemes:
        # arrays to contain the rho and r2 values
        rho_train_final, rho_test_final, r2_test_final = None, None, None

        logo = LeaveOneGroupOut() # leave one run out !
        for train, test in logo.split(fmri_runs, groups=range(4)):
            fmri_data_train = np.vstack([fmri_runs[i] for i in train])

            directions_train = directions[:, train] # retrieve directions of training
            direction_train_indices = np.unique(
                    directions_train, axis=0, return_index=True)[1]

            rho_train_all = [None for i in range(8)]
            rho_test_all = [[None for j in range(2)] for i in range(8)]

            for k_dir_train in direction_train_indices : # loop over the directions
                # TRAINING
                predictor = design_matrices[k_scheme][k_dir_train]
                predictors_train = np.vstack([predictor.loc[predictor[
                        f'session{i+1}']==1] for i in train]) # only take train sessions

                model = LinearRegression().fit(predictors_train,
                                        fmri_data_train) # fit the model

                # store all rho for being able to identify the best one
                rho_train_all[k_dir_train] = pearsonr(fmri_data_train,
                                               model.predict(predictors_train))[0]

                # TESTING
                direction_test_indices = []
                for d in range(n_directions):
                    # get the index of the directions that are similar for the training but different for the test
                    if (np.array_equal(directions[d, train], directions[k_dir_train, train]) == True):
                        direction_test_indices.append(d)

                for k_dir_test in direction_test_indices:
                    test_run = test[0] # fmri session index kept for the test
                    predictor = design_matrices[k_scheme][k_dir_test] # load design matrix
                    predictor_test = predictor.loc[predictor[f'session{test_run+1}']==1]

                    rho_test_unique = pearsonr(fmri_runs[test_run], model.predict(predictor_test))[0]

                    if rho_test_all[k_dir_train] is None :
                        rho_test_all[k_dir_train] = rho_test_unique
                    else:
                        np.stack((rho_test_all[k_dir_train], rho_test_unique))

            rho_train_all = np.array(rho_train_all)
            rho_test_all = np.array(rho_test_all)

            rho_train = np.max(rho_train_all, axis=0) # final rho train map
            rho_test = np.zeros(fmri_data_train.shape[1]) # final rho test map with the size of the no. voxels
            rsquares_test = np.zeros(fmri_data_train.shape[1])

            for k_voxel in range(fmri_data_train.shape[1]):
                # Identify the best train direction to be kept for the test
                best_train_index = np.argwhere(rho_train_all[:,k_voxel]==np.amax(
                        rho_train_all[:,k_voxel]))
                best_test_index = np.argwhere(rho_test_all[best_train_index][:, k_voxel] == np.amax(
                        rho_test_all[best_train_index][:, k_voxel]))
                rho_test[k_voxel] = np.max(rho_test_all[best_train_index][best_test_index])

                '''TODO: redefine the model here to get the right weights, or save r2 before'''
                rsquares_test[k_voxel] = r2_score(fmri_runs[test_run],
                                                  model.predict(design_matrices[k_scheme][best_test_index]),
                                                  multioutput='raw_values')

            log(rsquares_training, rsquares_test)

            rho_train_final = rho_train if rho_train_final is None else np.vstack([rho_train_final, rho_train])
            rho_test_final = rho_test if rho_test_final is None else np.vstack([rho_test_final, rho_test])
            r2_test_final = rsquares_test if r2_test_final is None else np.vstack([r2_test_final, rsquares_test])

        return (np.mean(rho_train_final, axis=0), np.mean(rho_test_final, axis=0),
                np.mean(r2_test_final, axis=0)) # average for each scheme
     
#%%

def do_single_subject(rootdir, subj, matrices):
    fmri_filenames = get_fmri_files_from_subject(rootdir, subj)
    fmri_runs = [masker.transform(f) for f in fmri_filenames]

    loglabel = subj
    logcsvwriter = csv.writer(open("test.log", "a+"))
    
    r2train, r2test = compute_crossvalidated_r2(fmri_runs, matrices, loglabel, logcsvwriter)

    r2train_img = masker.inverse_transform(r2train)
    nib.save(r2train_img, f'train_{subj:02d}.nii.gz')

    r2test_img = masker.inverse_transform(r2test)
    nib.save(r2test_img, f'test_{subj:02d}.nii.gz')

    display = plot_glass_brain(r2test_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(f'test_{subj:02}.png')
    display.close()

    display = plot_glass_brain(r2train_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(f'train_{subj:02}.png')
    display.close()

#%%
if __name__ == '__main__':

    DEBUG = True

    ROOTDIR = "/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014"
    MATRIXDIR = "ENCODAGE/design_matrices"
    MATRICES = get_design_matrices(op.join(ROOTDIR, MATRIXDIR))
    
    schemes = ['gaussian_ppc_mu', 'gaussian_ppc_conf', 'sigmoid_ppc_mu', 'sigmoid_ppc_conf', 
               'gaussian_dpc', 'sigmoid_dpc', 'rate_mu', 'rate_conf']
    n_schemes = len(schemes)
    
    directions = np.array(list(itertools.product([0,1], repeat=4)))
    n_directions = len(schemes)
    
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21]

    if DEBUG:
        subjects = subjects[:2]  # test on first 2 subjects only

    print(f'Computing global mask...')
    masker = compute_global_masker(ROOTDIR)

    for sub in subjects:
        print(f'Processing subject {sub:02d}...')
        do_single_subject(ROOTDIR, sub, MATRICES)
