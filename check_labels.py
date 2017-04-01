import numpy as np
import cv2
import pickle
import matplotlib.image as mpimg

train_images = pickle.load(open( "road_images.p", "rb" ))
train_images = train_images[:1019]

labels = pickle.load(open( "lane_labels.p", "rb" ))
labels = labels[:1019]

perspective_M = pickle.load(open('perspective_matrix.p', "rb" ))
Minv = np.linalg.inv(perspective_M)

other = cv2.cvtColor(np.copy(train_images[0]), cv2.COLOR_RGB2GRAY)

lane_lines = []

for n in range(len(train_images)):
    left_fit = labels[n][0:3]
    right_fit = labels[n][3:]

    fity = np.linspace(0, other.shape[0]-1, other.shape[0])
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
    
    warp_zero = np.zeros_like(other).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (train_images[n].shape[1], train_images[n].shape[0]))

    result = cv2.addWeighted(train_images[n], 1, newwarp, 0.3, 0)
    
    mpimg.imsave("labelled//labelled%d.jpg" % (n), result)
