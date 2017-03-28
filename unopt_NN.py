import numpy as np
import cv2
import glob
import pickle
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
from numpy import newaxis

""" This file contains the unoptimized convolutional neural network
for detecting lane lines. This file currently contains the necessary code
to 1) load the perspective transformed images and labels pickle files,
2) normalize the third and sixth labels for the image width (note this is
one item to be updated in a more optimal network for the other labels), 
3) resizes and grayscales the images for more efficient training,
4) shuffles and then splits the data into training and validation sets,
5) creates the neural network architecture, 6) trains the network,
7) saves down the model architecture and weights, 8) shows the model summary.
"""

# Load training images
train_images = pickle.load(open( "perspective_images.p", "rb" ))

# Load image labels
labels = pickle.load(open( "lane_labels.p", "rb" ))

def normalize(num):
    """ Normalizes the value of the constant within the label
    for each line, based on the image width of 1280 pixels.
    """
    new_num = (num - 640) / 1280
    return new_num

def norm_label(label):
    """ Feeds each of the lane line constants into
    the normalize function above for a given label.
    """
    label[2] = normalize(label[2])
    label[5] = normalize(label[5])
    return label

# Iterate through all labels to normalize them.
for n in range(len(labels)):
    labels[n] = norm_label(label)

""" The below iterates through all images to resize down
to 1/8th size, grayscales, and then adds an additional
dimension back to make feeding into the network easier.
Note that the grayscaling removes the extra dimension.
This should be a function, but the eventual plan is to
do the below in a python generator.
"""
for n, i in enumerate(train_images):
    new_image = imresize(train_images[n], (90, 160, 3))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    new_image = new_image[..., newaxis]
    train_images[n] = new_image

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)
# Test size may be 10% or 20%; random_state is set just to see how changing paramaters affects output
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1, random_state=23)

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 100
epochs = 100
pool_size = (2, 2)
input_shape = X_train.shape[1:]

# Here is the actual neural network
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# NOTE: Changing # of layers, type of layers, and all inputs to the layers are fair game for optimization
# Conv Layer 1
model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Conv Layer 2
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Flatten and Dropout
model.add(Flatten())
model.add(Dropout(0.5))

# FC Layer 1
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC Layer 2
model.add(Dense(32))
model.add(Activation('relu'))

# FC Layer 3
model.add(Dense(16))
model.add(Activation('relu'))

# Final FC Layer - six outputs - the three coefficients for each of the two lane lines polynomials
model.add(Dense(6))


# Compiling and training the model
# Currently using MAE instead of MSE for loss due to a few erratic labels that need fixing
model.compile(metrics=['accuracy'], optimizer='Adam', loss='mean_absolute_error')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# Show summary of model
model.summary()
