"""For loading camera calibration images.
Get at least 15 chessboard images with your camera.
Note that iPhones use a different size for images vs.
video, so you may need to extract images from video of a chessboard.
"""

import numpy as np
import os
import cv2
import pickle

# Load calibration images
loc = 'iphone_cam_cal/'
calibration_pics_loc = os.listdir(loc)
calibration_images = []

for i in calibration_pics_loc:
    i = loc + i
    image = cv2.imread(i)
    calibration_images.append(image)

# Prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays for later storing object points and image points
objpoints = []
imgpoints = []

for image in calibration_images:
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        cv2.drawChessboardCorners(image, (9, 6), corners, ret)

#Get undistortion info and undistort
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
dst = cv2.undistort(calibration_images[0], mtx, dist, None, mtx)

# Saving undistortion info
pickle.dump(mtx,open( "mtx.p", "wb" ))
pickle.dump(dist,open( "dist.p", "wb" ))
