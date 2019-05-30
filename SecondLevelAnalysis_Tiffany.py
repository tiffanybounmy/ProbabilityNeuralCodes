#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code to perform the second level analysis:
    - load the subject rho maps
    - compute paired differences
    - compute group-level t-test, and save the z-map on disk

@author: Florent Meyniel
"""

#%%
import glob
import os
import pandas as pd
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, mean_img, threshold_img
from nilearn import datasets, surface, plotting
import nibabel as nib
from nistats.second_level_model import SecondLevelModel
from scipy.stats import norm
from nilearn import plotting

#%%
# Define directories
rootdir = '/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014/'
first_dir = 'ENCODAGE/first_level_analyses'
second_dir = 'ENCODAGE/second_level_analyses'

#%%
def get_fmri_files(rootdir, model_name):
    return sorted(glob.glob(os.path.join(rootdir,
                                         f"subj*_{model_name}_test_rho.nii.gz")))

#%% 

def compute_global_masker(rootdir, FWHM=None):
    '''
    Define the mask that will be applied onto fMRI data
    '''
    mask = os.path.join(rootdir, 'spm12/tpm/mask_ICV.nii')
    global_mask = math_img('img>0', img=mask)
    masker = NiftiMasker(mask_img=global_mask, smoothing_fwhm=FWHM)
    masker.fit()
    
    return masker

#%%
def compute_model_difference(model1, model2):
    """
    Compute, for each subject, the (masked) difference between two models and 
    save the result on disk
    """
    masker = compute_global_masker(rootdir, FWHM=5)
    
    fmri_model_1 = [masker.transform(f) for f in get_fmri_files(os.path.join(rootdir,
                    first_dir), model1)]
    fmri_model_2 = [masker.transform(f) for f in get_fmri_files(os.path.join(rootdir, 
                    first_dir), model2)]
    
    for subj, (data1, data2) in enumerate(zip(fmri_model_1, fmri_model_2)):
        nib.save(masker.inverse_transform(data1 - data2), 
                 os.path.join(rootdir, first_dir, "diff",
                              f"subj{(subj+1):02d}_{model1}-{model2}_test_rho.nii.gz"))
        
#%%
def compute_second_level(model1, model2, FWHM=None):
    """
    Compute the group-level significance of the paired difference between two
    models
    """
    # # compute and save the difference of two models
    # compute_model_difference(model1, model2)   # compute with the FirstLevelAnalysis code
    
    # redefine the mask, without smoothing
    masker = compute_global_masker(rootdir)
    
    # use those different files as input for group-level analysis
    second_level_input = get_fmri_files(os.path.join(rootdir, first_dir, "diff"),
                                        f'{model1}-{model2}')
    
    # prepare second level analysis (one sample t-test)
    design_matrix = pd.DataFrame([1] * len(second_level_input),
                                 columns=['intercept'])
    second_level_model = SecondLevelModel(masker)
    second_level_model = second_level_model.fit(second_level_input,
                                                design_matrix=design_matrix)
    
    # estimation the contrast 
    z_map = second_level_model.compute_contrast(output_type='z_score')
    # save to disk
    nib.save(z_map, os.path.join(rootdir, second_dir, f"GroupLevel_{model1}-{model2}"))
    
    # Get the map of positive values only
    z_val = masker.transform(z_map)
    z_val_pos = [val if val > 0 else 0 for val in z_val[0]]
    z_map_pos = masker.inverse_transform(z_val_pos)
    
    return z_map, z_map_pos

#%%
# Compute the second-level analysis for a pair of models
model1 = 'rate_mu'
model2 = 'baseline'
z_map, z_map_pos = compute_second_level(model1, model2)

#%%
# get surface mesh (fsaverage)
fsaverage = datasets.fetch_surf_fsaverage()

#%%
# project volume onto surface
# Caution: the projection is done for one hemisphere only, make sure to plot
# (below) for the same hemisphere
img_to_plot = surface.vol_to_surf(z_map_pos, fsaverage.pial_right)

#%%
p_val = 0.05
p_unc = norm.isf(p_val)

#%%
# plot surface
# note: the bg_map and hemi should be the same hemisphere as fsaverage.???
plotting.plot_surf_stat_map(fsaverage.infl_right, img_to_plot, 
                            hemi='right', bg_map=fsaverage.sulc_right,
                            view='lateral', 
                            title='Surface plot', colorbar=True,
                            threshold=1.65)

#%%
# Plot glass brain of the z_map
display = plotting.plot_glass_brain(
    z_map, threshold=p_unc, colorbar=True, display_mode='lyr', plot_abs=False,
    title=f"group {model1}-{model2} (unc p<{p_val})")
#display.savefig(os.path.join(rootdir, "brain_plots", f"group_{model1}_{model2}.png"))

#%%
# Plot glass brain of the z_map (only positive values)
display_pos = plotting.plot_glass_brain(
        z_map_pos, threshold=10, colorbar=True, display_mode='lyr', plot_abs=False,
        title=f"group {model1}-{model2} (unc p<{p_val})")

#%% check for the right intraparietal sulcus 
p_val = 0.05
p_unc = norm.isf(p_val)
display_stat_map = plotting.plot_stat_map(
        z_map_pos, threshold=p_unc, colorbar=True, cut_coords=(32, -68, 59),
        title = f"group {model1}-{model2} (unc p<{p_val})")

#%% check for the right inferior temporal gyrus 
p_val = 0.05
p_unc = norm.isf(p_val)
display_stat_map = plotting.plot_stat_map(
        z_map_pos, threshold=p_unc, colorbar=True, cut_coords=(56, -46, -14),
        title = f"group {model1}-{model2} (unc p<{p_val})")