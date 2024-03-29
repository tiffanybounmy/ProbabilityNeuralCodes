#! /usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Code to perform the fitting of the null model to real data, with a cross validation approach.
The output is a rho-map (Pearson's correlation for each voxel), that can be projected onto the cortical surface.
The input is the design matrix generated by the code 'BaselineDesignMatrixCreation.py'.

@author: Tiffany Bounmy

"""

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
def get_fmri_files_from_subject(rootdir, sub):
    '''Define the path of fMRI data for each subject'''
    return sorted(glob.glob(os.path.join(rootdir,
                                         "MRI_data/raw_data",
                                         "subj%02d" % sub,
                                         "fMRI/wtraepi*.nii")))

#%%
def get_baseline_design_matrix(modeldir, sub):
    '''Import the baseline design matrix for each subject'''
    dmtx_path = (op.join(modeldir,
                         "subj%02d" % sub,
                         'dmtx_baseline.pkl'))
    matrices = pd.read_pickle(dmtx_path)
    return matrices

#%%

def compute_global_masker(rootdir):
    '''Define the mask that will be applied onto data'''
    mask = op.join(rootdir, 'spm12/tpm/mask_ICV.nii')
    global_mask = math_img('img>0', img= mask)
    masker = MultiNiftiMasker(global_mask, smoothing_fwhm = 1.5,
                              high_pass = 1/128, t_r = 2,
                              detrend=True, standardize=True)
    masker.fit()
    return masker

#%%
def compute_crossvalidated_scores(fmri_runs, dmtx, loglabel, logcsvwriter):

    '''Perform 4-fold cross-validation'''

    def log(r2_train, rho_test):
        """ log stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, 'train', np.mean(r2_train), np.std(r2_train),
                       np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, 'test', np.mean(rho_test), np.std(rho_test),
                               np.min(rho_test), np.max(rho_test)])
    
    r2_train_final = None
    rho_test_final = None
    
    logo = LeaveOneGroupOut() # leave one run out
    for train, test in logo.split(fmri_runs, groups=range(4)):
        # Train 
        fmri_data_train = np.vstack([fmri_runs[i] for i in train])
        predictors_train = np.vstack([dmtx.loc[dmtx[
                f'session{i+1}']==1] for i in train])
        model = LinearRegression().fit(predictors_train, fmri_data_train)

        r2_train = r2_score(fmri_data_train,
                            model.predict(predictors_train),
                            multioutput='raw_values')

        # Test
        test_run = test[0]
        predictor_test = dmtx.loc[dmtx[f'session{test_run+1}']==1]
        fmri_pred_test = model.predict(predictor_test)
        rho_test = np.zeros(fmri_pred_test.shape[1], dtype=float)
        for ivox in range(fmri_pred_test.shape[1]):
            rho_test[ivox] = \
            pearsonr(fmri_runs[test_run][:,ivox], fmri_pred_test[:,ivox])[0]

        log(r2_train, rho_test) # save scores in a csv file

        r2_train_final = r2_train if r2_train_final is None else np.vstack([r2_train_final, r2_train])
        rho_test_final = rho_test if rho_test_final is None else np.vstack([rho_test_final, rho_test])
    
    return {'r2_train': np.mean(r2_train_final, axis=0),
            'rho_test': np.mean(rho_test_final, axis=0)}

#%%

def do_single_subject(ROOTDIR, OUTDIR, subj):
    fmri_filenames = get_fmri_files_from_subject(ROOTDIR, sub)
    print('fmri filenames defined!')
    dmtx = get_baseline_design_matrix(op.join(ROOTDIR, MODELDIR), sub) # baseline design matrix
    print('matrices have been imported!')
    fmri_runs = [masker.transform(f) for f in fmri_filenames]
    print('masked fmri data have been imported!')

    loglabel = subj
    logcsvwriter = csv.writer(open("res_baseline.log", "a+"))
    
    results = compute_crossvalidated_scores(fmri_runs, dmtx, loglabel, logcsvwriter)

    r2_train_img = masker.inverse_transform(results['r2_train'])
    nib.save(r2_train_img, op.join(OUTDIR, f'subj{sub:02d}_baseline_train_r2.nii.gz'))
    
    rho_test_img = masker.inverse_transform(results['rho_test'])
    nib.save(rho_test_img, op.join(OUTDIR, f'subj{sub:02d}_baseline_test_rho.nii.gz'))

    display = plot_glass_brain(r2_train_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(op.join(OUTDIR, "brain_plots", f'subj{sub:02d}_baseline_train_r2.png'))
    display.close()
    
    display = plot_glass_brain(rho_test_img, display_mode='lzry', threshold=0, colorbar=True)
    display.savefig(op.join(OUTDIR, "brain_plots", f'subj{sub:02d}_baseline_test_rho.png'))
    display.close()

#%%
if __name__ == '__main__':

    DEBUG = False

    ROOTDIR = "/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014"
    MODELDIR = "ENCODAGE/design_matrices"
    OUTDIR= "/volatile/bounmy/first_level_analyses"
    
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21]

    if DEBUG:
        subjects = subjects[:2]  # test on first 2 subjects only

    print(f'Computing global mask...')
    masker = compute_global_masker(ROOTDIR)

    for sub in subjects:
        print(f'Processing subject {sub:02d}...')
        do_single_subject(ROOTDIR, OUTDIR, sub)

