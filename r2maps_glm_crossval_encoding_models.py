#! /usr/bin/env python3
# Time-stamp: <>
#%%
import glob
import os
import os.path as op
# import sys
import csv
import pandas as pd
import seaborn as sns
import itertools

import nibabel as nib
# from nilearn.masking import compute_epi_mask
# from nilearn.masking import apply_mask
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
import multiprocessing

#%%
def get_fmri_files_from_subject(rootdir, sub):
    return sorted(glob.glob(os.path.join(rootdir,
                                         "MRI_data/raw_data",
                                         "subj%02d" % sub,
                                         "fMRI/wtraepi*.nii")))
    
#%%
def get_design_matrices(modeldir, sub):
    # initialise the dictionary that will contain all design matrices for one sub
    matrices = {'gaussian_ppc_mu':[], 'gaussian_ppc_conf':[],
                'sigmoid_ppc_mu':[], 'sigmoid_ppc_conf':[],
                'gaussian_dpc':[], 'sigmoid_dpc':[],
                'rate_mu':[], 'rate_conf':[]}
    
    # get the path of the design matrices for one sub
    dmtx_path = sorted(glob.glob(op.join(modeldir, 
                                         "subj%02d" % sub,
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
    '''
    Compute masker for data
    '''
    mask = op.join(rootdir, 'spm12/tpm/mask_ICV.nii')
    global_mask = math_img('img>0', img= mask)
    masker = MultiNiftiMasker(global_mask, smoothing_fwhm = 1.5,
                              high_pass = 1/128, t_r = 2,
                              detrend=True, standardize=True)
    masker.fit()
    return masker

#%%
def compute_correlation(fmri_runs, fmri_pred_test, 
                        test_run, ivox):
    """
    Function used to parallelize the correlation across voxels
    """
    return pearsonr(fmri_runs[test_run][:,ivox], fmri_pred_test[:,ivox])[0]
    
#%%
def compute_crossvalidated_r2(fmri_runs, matrices, k_scheme,
                              loglabel, logcsvwriter):

    def log(r2_train, rho_test):
        """ log stats per fold to a csv file """
        logcsvwriter.writerow([loglabel, k_scheme, 'train', np.mean(r2_train), np.std(r2_train),
                               np.min(r2_train), np.max(r2_train)])
        logcsvwriter.writerow([loglabel, k_scheme, 'test', np.mean(rho_test), np.std(rho_test),
                               np.min(rho_test), np.max(rho_test)])
    
    #for k_scheme in schemes:
    #k_scheme = schemes[0]

    r2_train_final = None
    rho_test_final = None
    
    logo = LeaveOneGroupOut() # leave one run out !
    for train, test in logo.split(fmri_runs, groups=range(4)):
        fmri_data_train = np.vstack([fmri_runs[i] for i in train])
        
        # For the confidence encoding schemes, only consider 1 direction for train and test
        if k_scheme.find('conf') != -1:
            directions_train_indices = [0]
        # For the other schemes, select directions of training and the corresponding
        # directions of testing
        else:
            directions_train = directions[:, train] # retrieve directions of training
            directions_train_indices = np.unique(
                    directions_train, axis=0, return_index=True)[1]
        
        # initialise r2 train for all directions of training
        r2_train_all = [None for k_dir_train in range(8)]
        # initialise rho test for all directions of train and test
        rho_test_all = [[None for k_dir_test in range(2)] for k_dir_train in range(8)]
        
        for i_train in range(len(directions_train_indices)) : # loop over the directions 
            # TRAINING 
            k_dir_train = directions_train_indices[i_train]
            predictor = matrices[k_scheme][k_dir_train]
            predictors_train = np.vstack([predictor.loc[predictor[
                    f'session{i+1}']==1] for i in train]) # only take train sessions
            
            # fit the model
            model = LinearRegression().fit(predictors_train,
                                    fmri_data_train) # fit the model
            
            # compute the r2 scores for all voxels 
            r2_train_all[i_train] = r2_score(fmri_data_train, 
                                     model.predict(predictors_train),
                                     multioutput='raw_values') 
                       
            # TESTING
            if k_scheme.find('conf') != -1:
                direction_test_indices = [0]
            else:
                direction_test_indices = []
                for d in range(n_directions):
                    if (np.array_equal(directions[d, train], directions[k_dir_train, train]) == True):
                        direction_test_indices.append(d)
            
            for i_test in range(len(direction_test_indices)):
                k_dir_test = direction_test_indices[i_test]
                test_run = test[0] # fmri session index kept for the test
                predictor = matrices[k_scheme][k_dir_test] # load design matrices
                predictor_test = predictor.loc[predictor[f'session{test_run+1}']==1]
                
                fmri_pred_test = model.predict(predictor_test)
                
                # compute rho with a loop over voxels (do pearsonr for eah voxel separately)
                rho_test_unique = np.array(
                        Parallel(n_jobs=n_jobs)(delayed(compute_correlation)(
                                fmri_runs, fmri_pred_test, test_run, ivox) \
                                                    for ivox in range(fmri_runs[test_run].shape[1])))
                
                rho_test_all[i_train][i_test] = rho_test_unique
        
        
        if k_scheme.find('conf') != -1:
            r2_train = r2_train_all[0]
            rho_test = rho_test_all[0][0]
        else:
            r2_train_all = np.array(r2_train_all) # convert in order to retrieve the best index
            rho_test_all = np.array(rho_test_all) # this one is the rho for test
            r2_train = np.max(r2_train_all, axis=0) # final r2 (train) map
            rho_test = np.zeros(fmri_data_train.shape[1]) # final rho (test) map
            
            for k_vox in range(fmri_data_train.shape[1]):
                # Identify the best train direction to be kept for the test
                best_train_index = np.argmax(r2_train_all[:,k_vox])
                # Identify the best direction of test
                best_test_index = np.argmax(rho_test_all[best_train_index,:,k_vox])
                # Retrieve the best rho map
                rho_test[k_vox] = rho_test_all[best_train_index, best_test_index, k_vox]
    
        print('crossval done for the fold!')       
        log(r2_train, rho_test)

        r2_train_final = r2_train if r2_train_final is None else np.vstack([r2_train_final, r2_train])
        rho_test_final = rho_test if rho_test_final is None else np.vstack([rho_test_final, rho_test])
    
    return {'r2_train':np.mean(r2_train_final, axis=0), 
            'rho_test':np.mean(rho_test_final, axis=0)} # average for each scheme 
     
#%%

def do_single_subject(ROOTDIR, OUTDIR, subj):
    fmri_filenames = get_fmri_files_from_subject(ROOTDIR, sub)
    print('fmri filenames defined!')
    matrices = get_design_matrices(op.join(ROOTDIR, MODELDIR), sub)
    print('matrices have been imported!')
    fmri_runs = [masker.transform(f) for f in fmri_filenames]
    print('masked fmri data have been imported!')

    loglabel = subj
    
    for k_scheme in schemes:
        logcsvwriter = csv.writer(open(f'first_level_analyses.log', "a+"))
        results = compute_crossvalidated_r2(fmri_runs, matrices, k_scheme, 
                                                       loglabel, logcsvwriter)

        r2_train_img = masker.inverse_transform(results['r2_train'])
        nib.save(r2_train_img, op.join(OUTDIR,f'subj{sub:02d}_{k_scheme}_train_r2.nii.gz'))
        
        rho_test_img = masker.inverse_transform(results['rho_test'])
        nib.save(rho_test_img, op.join(OUTDIR,f'subj{sub:02d}_{k_scheme}_test_rho.nii.gz'))
    
        display = plot_glass_brain(r2_train_img, display_mode='lzry', threshold=0, colorbar=True)
        display.savefig(op.join(OUTDIR, "brain_plots", f'subj{sub:02d}_{k_scheme}_train_r2.png'))
        display.close()
    
        display = plot_glass_brain(rho_test_img, display_mode='lzry', threshold=0, colorbar=True)
        display.savefig(op.join(OUTDIR, "brain_plots", f'subj{sub:02d}_{k_scheme}_test_rho.png'))
        display.close()

#%%
if __name__ == '__main__':

    DEBUG = False

    ROOTDIR = "/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014"
    MODELDIR = "ENCODAGE/design_matrices"
    OUTDIR = "/volatile/bounmy/first_level_analyses"
    
    schemes = ['gaussian_ppc_mu', 'gaussian_ppc_conf', 'sigmoid_ppc_mu', 'sigmoid_ppc_conf', 
               'gaussian_dpc', 'sigmoid_dpc', 'rate_mu', 'rate_conf']
    n_schemes = len(schemes)
    
    directions = np.array(list(itertools.product([0,1], repeat=4)))
    n_directions = len(directions)
    
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                19, 20, 21]

    # options for parallel processing
    num_cores = multiprocessing.cpu_count()
    n_jobs = int(num_cores/2)
    
    if DEBUG:
        subjects = subjects[:2]  # test on first 2 subjects only

    print(f'Computing global mask...')
    masker = compute_global_masker(ROOTDIR)

    for sub in subjects:
        print(f'Processing subject {sub:02d}...')
        do_single_subject(ROOTDIR, OUTDIR, sub)