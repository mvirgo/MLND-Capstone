""" This file uses the original polynomial coefficient labels
in order to create an image with a green lane to overlay on
the original road image. The file originally saved down any
re-combined images in order to check whether the labels were
appropriate; the current version saves them all to a pickle file
for usage as the true labels for the fully convolutional neural
network for training.
"""

import numpy as np
import cv2
import pickle
from scipy.misc import imresize

# Load all training images
train_images = pickle.load(open( "good_road_images.p", "rb" ))

# Load all polynomial coefficient labels
labels = pickle.load(open( "lane_labels.p", "rb" ))

# Load the perspective transformation matrix, and calculate the inverse
perspective_M = pickle.load(open('perspective_matrix.p', "rb" ))
Minv = np.linalg.inv(perspective_M)

other = cv2.cvtColor(np.copy(train_images[0]), cv2.COLOR_RGB2GRAY)

lane_lines = []

for n in range(len(train_images)):
    # Left line is first three coefficients, right line is the other three
    left_fit = labels[n][0:3]
    right_fit = labels[n][3:]

    # Fit polynomials
    fity = np.linspace(0, other.shape[0]-1, other.shape[0])
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

    # Create a space to draw the lane image labels
    warp_zero = np.zeros_like(other).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Points to fill on the lane image label
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill the space
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    # Warp back to the non-bird's eye view of the road
    newwarp = cv2.warpPerspective(color_warp, Minv, (train_images[n].shape[1], train_images[n].shape[0]))

    # Using 80x160x3 images for training; also only needs 'G' channel
    # Lane is drawn in 'G' so the others are unneeded
    newwarp = imresize(newwarp, (80, 160, 3))
    newwarp = newwarp[:,:,1]
    newwarp = newwarp[:,:,None]

    lane_lines.append(newwarp)

# Save labels to pickle file
pickle.dump(lane_lines,open('image_lane_labels.p', "wb" ))
