#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:11:36 2019
Example script showing how to project on the surface
@author: Florent Meyniel
"""

import glob
import os
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, mean_img, threshold_img
import nibabel as nib
from nilearn import datasets, surface, plotting
from scipy.stats import norm

#%%
# Define directories
rootdir = '/neurospin/unicog/protocols/IRMf/Meyniel_MarkovGuess_2014'
first_dir = 'ENCODAGE/first_level_analyses'
second_dir = 'ENCODAGE/second_level_analyses'

#%%
# Compute masker for data
mask = os.path.join(rootdir, 'spm12/tpm/mask_ICV.nii')
global_mask = math_img('img>0', img=mask)
masker = NiftiMasker(mask_img=global_mask, smoothing_fwhm=None)
masker.fit()

#%%
## Plot single subject
# load smoothed data, when convert to nifti object
model1 = 'sigmoid_dpc'
model2 = 'baseline'
file = glob.glob(os.path.join(rootdir, first_dir, "diff",
                              f'subj01_{model1}-{model2}_test_rho.nii.gz'))
rho_map = masker.transform(file)
rho_map_pos = [val if val > 0 else 0 for val in rho_map[0]]

rho_map = masker.inverse_transform(rho_map)
rho_map_pos = masker.inverse_transform(rho_map_pos)

#%%
# get surface mesh (fsaverage)
fsaverage = datasets.fetch_surf_fsaverage()

#%%
# project volume onto surface
# Caution: the projection is done for one hemisphere only, make sure to plot
# (below) for the same hemisphere
img_to_plot = surface.vol_to_surf(rho_map_pos, fsaverage.pial_right)

#%%
# plot surface
# note: the bg_map and hemi should be the same hemisphere as fsaverage.???
plotting.plot_surf_stat_map(fsaverage.infl_right, img_to_plot,
                            hemi='right', bg_map=fsaverage.sulc_right,
                            view='lateral',
                            title='Surface plot', colorbar=True,
                            threshold=0.01)

#%%
# plot glass brain
plotting.plot_glass_brain(rho_map_pos, display_mode='lyrz', plot_abs=False,
                          title='Glass brain', threshold=0.005)

#%%
## Plot Group
p_val = 0.05
p_unc = norm.isf(p_val)

z_file = os.path.join(rootdir, second_dir, f'GroupLevel_{model1}-{model2}.nii')
z_map = masker.transform(z_file)
z_map_pos = [val if val > 0 else 0 for val in z_map[0]]
z_map = masker.inverse_transform(z_map)
z_map_pos = masker.inverse_transform(z_map_pos)
z_map_img = surface.vol_to_surf(z_map_pos, fsaverage.pial_right)

plotting.plot_surf_stat_map(fsaverage.infl_right, z_map_img,
                            hemi = 'right', bg_map=fsaverage.sulc_right,
                            view='medial',
                            title=f'Surface plot {model1} - {model2} (unc p<{p_val}) ', colorbar=True,
                            threshold=p_unc)

#%%
# check for the RIPS
display_stat_map = plotting.plot_stat_map(
        z_map_pos, threshold=p_unc, colorbar=True, cut_coords=(32, -68, 59),
        title = f"group {model1}-{model2} (unc p<{p_val})")

#%%
# check for the RIG
display_stat_map = plotting.plot_stat_map(
        z_map_pos, threshold=p_unc, colorbar=True, cut_coords=(56, -46, -14),
        title = f"group {model1}-{model2} (unc p<{p_val})")

#%%
# Check for the stat map
display_stat_map = plotting.plot_stat_map(
        z_map_pos, threshold=p_unc, colorbar=True,
        title = f"group {model1}-{model2} (unc p<{p_val})")