#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to perform the second level analysis:
    - load the subject rho maps
    - compute paired differences
    - compute group-level t-test, and save the z-map on disk

@author: Florent Meyniel
"""

import glob
import os
import pandas as pd
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, mean_img, threshold_img
import nibabel as nib
from nistats.second_level_model import SecondLevelModel
from scipy.stats import norm
from nilearn import plotting

# Define directories
rootdir='/home/fm239804/Dropbox/tmp_data'
spm_dir = '/home/fm239804/toolbox/matlab'

def get_fmri_files(rootdir, model_name):
    return sorted(glob.glob(os.path.join(rootdir,
                                         f"subj*_{model_name}.nii.gz")))

def compute_model_difference(model1, model2, masker):
    """
    Compute, for each subject, the (masked) difference between two models and 
    save the result on disk
    """
    fmri_model_1 = [masker.transform(f) for f in get_fmri_files(rootdir, model1)]
    fmri_model_2 = [masker.transform(f) for f in get_fmri_files(rootdir, model2)]
    
    for subj, (data1, data2) in enumerate(zip(fmri_model_1, fmri_model_2)):
        nib.save(masker.inverse_transform(data1 - data2), 
                 os.path.join(rootdir, f"subj{(subj+1):02d}_diff.nii.gz"))

def compute_second_level(model1, model2, FWHM=None):
    """
    Compute the group-level significance of the paired difference between two
    models
    """
    # Compute masker for data
    mask = os.path.join(spm_dir, 'spm12/tpm/mask_ICV.nii')
    global_mask = math_img('img>0', img=mask)
    masker = NiftiMasker(mask_img=global_mask)
    masker.fit()

    # compute and save the difference of two models
    compute_model_difference(model1, model2, masker)
    
    # use those difference files as input for group-level analysis
    second_level_input = get_fmri_files(rootdir, 'diff')
    
    # prepare second level analysis (one sample t-test)
    design_matrix = pd.DataFrame([1] * len(second_level_input),
                                 columns=['intercept'])
    second_level_model = SecondLevelModel(masker, smoothing_fwhm=FWHM)
    second_level_model = second_level_model.fit(second_level_input,
                                                design_matrix=design_matrix)
    
    # estimation the contrast 
    z_map = second_level_model.compute_contrast(output_type='z_score')
    
    # save to disk
    nib.save(z_map, os.path.join(rootdir, f"GroupLevel_{model1}-{model2}"))
    return z_map

# Compute the second-level analysis for a pair of model
# Ultimately: do it for all models
model1 = 'GaussianDPC'
model2 = 'null'
z_map = compute_second_level(model1, model2, FWHM=2)

# Threshold (uncorrected) the second level contrast and plot
p_val = 0.5
p_unc = norm.isf(p_val)
display = plotting.plot_glass_brain(
    z_map, threshold=p_unc, colorbar=True, display_mode='lyr', plot_abs=False,
    title=f"group {model1}-{model2} (unc p<{p_val})")
