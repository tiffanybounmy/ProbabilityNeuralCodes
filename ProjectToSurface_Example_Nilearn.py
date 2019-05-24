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

# Define directories
rootdir='/home/fm239804/tmp'
spm_dir = '/home/fm239804/toolbox/matlab'

# Compute masker for data
mask = os.path.join(spm_dir, 'spm12/tpm/mask_ICV.nii')
global_mask = math_img('img>0', img=mask)
masker = NiftiMasker(mask_img=global_mask, smoothing_fwhm=5)
masker.fit()

# load and smooth, when convert to nifti object
file = glob.glob(os.path.join(rootdir, 'subj01_diff_test_rho.nii.gz'))
rho_map = masker.transform(file)
rho_map = masker.inverse_transform(rho_map)

# get surface mesh(fsaverage)
fsaverage = datasets.fetch_surf_fsaverage()

# project volume onto surface
# Caution: the projection is done for one hemisphere only, make sure to plot
# (below) for the same hemisphere
img_to_plot = surface.vol_to_surf(rho_map, fsaverage.pial_right)

# plot surface
# note: the bg_map and hemi should be the same hemisphere as fsaverage.???
plotting.plot_surf_stat_map(fsaverage.infl_right, img_to_plot, 
                            hemi='right', bg_map=fsaverage.sulc_right,
                            view='medial', 
                            title='Surface plot', colorbar=True,
                            threshold=0.005)

# plot glass brain
plotting.plot_glass_brain(rho_map, display_mode='lyrz', plot_abs=False,
                          title='Glass brain', threshold=0.005)