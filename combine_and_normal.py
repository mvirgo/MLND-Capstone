''' This file merges my image files and label files.
It also resizes the training images. The file also
generates additional image data based on images outside
the very middle of the original polynomial coefficient
distributions in three waves, and adds rotations to both the
road image and corresponding lane image label. Also, for all
road images, an extra is added by horizontally flipping both
the image and its lane image label.
'''

import numpy as np
import cv2
import pickle
from scipy.misc import imresize
from scipy import ndimage

def find_to_rotate(labels, ignore_range):
    """Finds the label numbers in the loaded data which
    fall outside a given range.
    """
    # List for each coefficient
    labels_by_coeff = [[],[],[],[],[],[]]

    # Iterate through each label and append to list
    for label in labels:
        for n in range(len(label)):
            labels_by_coeff[n].append(label[n])

    images_to_rotate = []

    # Append to the list for all outside of the ranges above
    for n in range(len(labels)):
        for x in range(len(labels[0])):
            if labels[n][x] < ignore_range[x][0] or labels[n][x] > ignore_range[x][1]:
                images_to_rotate.append(n)

    # Make the list into unique values only (i.e. only one for each image)
    images_to_rotate = np.unique(images_to_rotate)

    return images_to_rotate

def rotate_images(train_images, pic_labels, images_to_rotate, angles):
    """ Adds "fake" training data using rotations of the original
    road images and lane image labels.
    """
    # Small angles to rotate the images
    more_X = []
    more_y = []

    # Create new images (with same label as original) with the rotation angles above
    for n in (images_to_rotate):
        for l in range(len(angles)):
            more_X.append(ndimage.rotate(train_images[n], angles[l], reshape = False))
            more_y.append(ndimage.rotate(pic_labels[n], angles[l], reshape = False))

    # Combine the old training images and labels with the new data
    train_images = train_images + more_X
    pic_labels = pic_labels + more_y

    return train_images, pic_labels

def horiz_flip(train_images, pic_labels):
    """Adds additional "fake" training data by horizontally flipping
    the road images and lane image labels, after additional were already added
    by the image rotations.
    """
    # Hold images prior to appending at once
    more_X = []
    more_y = []

    # Flip images horizontally - note that doing so in the keras generator would not flip labels
    for i in range(len(train_images)):
        more_X.append(np.fliplr(train_images[n]))
        more_y.append(np.fliplr(pic_labels[n]))

    # Combine the old training images and labels with the new data
    train_images = train_images + more_X
    pic_labels = pic_labels + more_y

    return train_images, pic_labels

# Load in all training images
train_images = pickle.load(open( "good_road_images.p", "rb" ))
train_images_curve = pickle.load(open( "good_road_images_curve.p", "rb" ))
train_images_u = pickle.load(open( "good_u_road_images.p", "rb" ))

# Combine into one
train_images = train_images + train_images_curve + train_images_u

# Clear out memory for the unneeded data
train_images_curve = []
train_images_u = []

# Downsize training images
for n in range(len(train_images)):
    new_image = imresize(train_images[n], (80, 160, 3))
    train_images[n] = new_image

# Load in all labels (coefficients)
labels = pickle.load(open( "lane_labels.p", "rb" ))
labels_curve = pickle.load(open( "lane_labels_curve.p", "rb" ))
labels_u = pickle.load(open( "u_lane_labels.p", "rb" ))

# Combine into one
labels = labels + labels_curve + labels_u

# Clear out unneeded data
labels_curve = []
labels_u = []

# Load in all picture labels (redrawn lines in G color)
pic_labels = pickle.load(open( "image_lane_labels.p", "rb" ))
pic_labels_curve = pickle.load(open( "image_lane_labels_curve.p", "rb" ))
pic_labels_u = pickle.load(open( "image_u_lane_labels.p", "rb" ))

# Combine into one
pic_labels = pic_labels + pic_labels_curve + pic_labels_u

# Clear out unneeded data
pic_labels_curve = []
pic_labels_u = []

# Small angles to rotate the images
angles = [-1, 1, -2, 2]

# Ranges to ignore for adding additional rotations
# The first has roughly 400 per coeff outside the range, second ~100, and third ~30
fine_range = (((-.00067,.000383),(-0.99,0.73),(-35,650),(-.00052,.00058),(-1.06,0.75),(419,1240)),
              ((-.00130,.00165),(-2.1,1.8),(-290,1165),(-.0014,.00186),(-2.5,1.86),(-40,1700)),
              ((-.00193,.00270),(-3.59,2.46),(-578,1590),(-.002,.00298),(-3.7,3),(-270,2372)))

for n in range(len(fine_range)):
    images_to_rotate = find_to_rotate(labels, fine_range[n])
    train_images, pic_labels = rotate_images(train_images, pic_labels, images_to_rotate, angles)

# Adds horizontal flips of each image and lane imagelabel
train_images, pic_labels = horiz_flip(train_images, pic_labels)

# Save images to pickle file
pickle.dump(train_images,open('full_CNN_train.p', "wb" ))
train_images = []

# Save labels to pickle file
pickle.dump(pic_labels,open('full_CNN_labels.p', "wb" ))
