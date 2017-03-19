"""This file undistorts each road image using the camera calibration
obtained from '2_cam_calib.py'. The images are then perspective
transformed to provide a top-down view.
"""

import numpy as np
import cv2
import matplotlib.image as mpimg
import pickle

# Loading undistortion info
mtx = pickle.load(open( "mtx.p", "rb" ))
dist = pickle.load(open( "dist.p", "rb" ))

# Re-load road images
road_images = pickle.load(open('road_images.p', "rb" ))

# Make list for undistorted images
undist_road = []

for image in road_images:
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    undist_road.append(dst)
    
road_images = []

# Perspective transform
def birds_eye(img, mtx, dist):
    """ Performs perspective transformation on an image.
    The image will appear as if from above, like a bird's
    view from the sky looking toward the ground.
    """
    # Undistort the image using camera calibration info
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert to grayscale
    gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
    offset = 300 # offset for dst points
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])
    # Source points for the location of points to shift to new destinations
    src = np.float32([[755,425],[1200, 720],[80,720],[525, 425]])
    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp the image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M

# Perspective transform the undistorted images, then save them
# Note that the perspective matrix also needs to be saved
# This is based off the source and destination points above,
# so only one is needed. This will be used later to revert the images
# back to normal.

# Counter so perspective matrix only saved once
counter = 0

# Iterate through each image, perspective transform, save.
for n, i in enumerate(undist_road):
    top_down, perspective_M = birds_eye(i, mtx, dist)
    mpimg.imsave("perspect/perspect%d.jpg" % (n), top_down)
    counter += 1
    if counter == 1:
        pickle.dump(perspective_M,open('perspective_matrix.p', "wb" ))
