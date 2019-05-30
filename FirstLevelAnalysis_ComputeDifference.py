#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:54:36 2019

@author: tb258044
"""

#%%
import glob
import os
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img
import nibabel as nib

#%%
# Define directories
rootdir = '/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/ENCODAGE/first_level_analyses'
spm_dir = '/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014'

#%%
def get_fmri_files(rootdir, model_name):
    """
    Get files
    """
    return sorted(glob.glob(os.path.join(rootdir,
                                         f"subj*_{model_name}_test_rho.nii.gz")))

#%%
def compute_model_difference(model1, model2, masker):
    """
    Compute, for each subject, the (masked) difference between two models and
    save the result on disk
    """
    fmri_model_1 = [masker.transform(f) for f in get_fmri_files(rootdir, model1)]
    fmri_model_2 = [masker.transform(f) for f in get_fmri_files(rootdir, model2)]

    for subj, (data1, data2) in enumerate(zip(fmri_model_1, fmri_model_2)):
        nib.save(masker.inverse_transform(data1 - data2),
                 os.path.join(rootdir, f"subj{(subj+1):02d}_{model1}-{model2}_test_rho.nii.gz"))

#%%
def save_difference(model1, model2, FWHM=None):
    """
    Compute the group-level significance of the paired difference between two
    models
    """
    # Compute masker for data
    mask = os.path.join(spm_dir, 'spm12/tpm/mask_ICV.nii')
    global_mask = math_img('img>0', img=mask)
    masker = NiftiMasker(mask_img=global_mask, smoothing_fwhm=FWHM)
    masker.fit()

    # compute and save the difference of two models
    compute_model_difference(model1, model2, masker)

#%%
# Compute the second-level analysis for a pair of model
# Ultimately: do it for all models
model1 = 'rate_conf'
model2 = 'baseline'
save_difference(model1, model2, FWHM=5)
