""" This file merges my image files and label files, and 
normalizes the lane labels with StandardScaler. These are
saved so that the labels can later be  de-normalized after
prediction. It also adds rotations to the original images 
that are outside of the middle of the distribution of labels
(based off histograms by each label) to help the model 
generalize away from straight lines better.
"""

import numpy as np
import cv2
import pickle
from scipy.misc import imresize
from numpy import newaxis
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

# Load in all training images
train_images = pickle.load(open( "good_perspective_images.p", "rb" ))
train_images_curve = pickle.load(open( "good_perspective_images_curve.p", "rb" ))

# Combine into one
train_images = train_images + train_images_curve

# Clear out memory for the unneeded data
train_images_curve = []

# Load in all labels
labels = pickle.load(open( "lane_labels.p", "rb" ))
labels_curve = pickle.load(open( "lane_labels_curve.p", "rb" ))

# Combine into one
labels = labels + labels_curve

# Clear out unneeded data
labels_curve = []

# List for each coefficient
labels_by_coeff = [[],[],[],[],[],[]]

# Iterate through each label and append to list
for label in labels:
    for n in range(len(label)):
        labels_by_coeff[n].append(label[n])
        
# Below ranges are based on histograms of the distribution of the above
# These ignore the middle parts of the distribution to focus on rarer values
fine_range = ((-.00088,.000594),(-1.29,0.74),(-144,724),(-.00074,.00079),(-1.36,1),(304,1223))

images_to_rotate = []

# Append to the list for all outside of the ranges above
for n in range(len(labels)):
    for x in range(len(labels[0])):
        if labels[n][x] < fine_range[x][0] or labels[n][x] > fine_range[x][1]:
            images_to_rotate.append(n)

# Make the list into unique values only (i.e. only one for each image)
images_to_rotate = np.unique(images_to_rotate)

# Small angles to rotate the images
angles = [-1, 1, -2, 2]
more_X = []
more_y = []

# Create new images (with same label as original) with the rotation angles above
for n in (images_to_rotate):
    for l in range(len(angles)):
        more_X.append(ndimage.rotate(train_images[n], angles[l], reshape = False))
        more_y.append(labels[n])

# Combine the old training images and labels with the new data
train_images = train_images + more_X
labels = labels + more_y

# The below code will normalize the labels by each coefficient.
label_scaler = StandardScaler()
labels = label_scaler.fit_transform(labels)

# Downsize, grayscale and normalize training images
# Note that `newaxis` is needed so Keras receives the dimensions it expects
for n in range(len(train_images)):
    new_image = imresize(train_images[n], (45, 80, 3))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    new_image = new_image[..., newaxis]
    new_image = (new_image / 255) * .8 + 1
    train_images[n] = new_image

# Save images, labels, and scaler to pickle files
# Note that the scaler will be needed to revert predicted labels to normal
pickle.dump(train_images,open('full_perspect_train.p', "wb" ))
pickle.dump(labels,open('full_labels.p', "wb" ))
pickle.dump(label_scaler,open('scaler.p',"wb"))
