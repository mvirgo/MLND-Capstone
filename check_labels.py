""" This file pulls in road images (created by `load_road_images.py`
and respective labels (that have been created by `make_labels.py`). 
Using those labels, it draws on the lane area onto the image. This file 
is for the purpose of checking whether the labels appear fairly accurate,
or whether further manual manipulation is needed of the lines being fed 
to create the labels for a given image.
"""

import numpy as np
import cv2
import pickle
import matplotlib.image as mpimg

# Import road images and labels pickle files
train_images = pickle.load(open( "road_images.p", "rb" ))
labels = pickle.load(open( "lane_labels.p", "rb" ))

# Load the perspective transform (since the labels are on the transformed images)
perspective_M = pickle.load(open('perspective_matrix.p', "rb" ))
Minv = np.linalg.inv(perspective_M)

# Creates an image in a necessary shape for lane lines to be drawn before inverse transform
other = cv2.cvtColor(np.copy(train_images[0]), cv2.COLOR_RGB2GRAY)

lane_lines = []

# Iterate through each image, using labels to plot lane lines and draw the lane based on those boundaries
for n in range(len(train_images)):
    # Left line coefficients are first three labels, right line is last three
    left_fit = labels[n][0:3]
    right_fit = labels[n][3:]
    
    # Fit the labels to a polynomial equation for each line
    fity = np.linspace(0, other.shape[0]-1, other.shape[0])
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
    
    # Creates blank image to re-draw lines on
    warp_zero = np.zeros_like(other).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Points making up the lines
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Drawing the detected lane down based on the points
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    
    # Unwarps the detected lane from the bird's eye view to regular image view
    newwarp = cv2.warpPerspective(color_warp, Minv, (train_images[n].shape[1], train_images[n].shape[0]))
    
    # Draws the lane back onto the original road image
    result = cv2.addWeighted(train_images[n], 1, newwarp, 0.3, 0)
    
    # Save down the files (to a folder named 'labelled') for review of reasonableness of the lane detection
    mpimg.imsave("labelled//labelled%d.jpg" % (n), result)
