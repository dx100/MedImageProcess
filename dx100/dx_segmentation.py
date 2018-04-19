# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 13:45:37 2017

@author: dx100
"""

import os
import gzuidhof

INPUT_FOLDER = '/Users/dx100/Data/Kaggle/sample_images/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

case = gzuidhof.load_scan(INPUT_FOLDER + patients[1])
case_pixels= gzuidhof.get_pixels_hu(case)
    
pix_resampled, spacing = gzuidhof.resample(case_pixels, case, [1,1,1])

segmented_lungs = gzuidhof.segment_lung_mask(pix_resampled, False)
segmented_lungs_fill = gzuidhof.segment_lung_mask(pix_resampled, True)

from mayavi import mlab

mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

src = mlab.pipeline.scalar_field(segmented_lungs)
vol = mlab.pipeline.iso_surface(src, contours=[1,], color=(0.8, 0.7, 0.6))

