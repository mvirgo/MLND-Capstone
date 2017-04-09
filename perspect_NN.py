import numpy as np
import cv2
import glob
import pickle
import keras
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.misc import imresize

""" This file contains the convolutional neural network for detecting
lane lines on a perspective transformed image. This file currently contains
the necessary code to 1) load the perspective transformed images and labels
pickle files, 2) shuffles and then splits the data into training and validation
sets, 3) creates the neural network architecture, 4) trains the network,
5) saves down the model architecture and weights, 6) shows the model summary.
"""

# Load training images
train_images = pickle.load(open("full_perspect_train.p", "rb" ))

# Load image labels
labels = pickle.load(open("full_labels.p", "rb" ))

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)
# Test size may be 10% or 20%; random_state is set just to see how changing paramaters affects output
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 150
epochs = 10
pool_size = (2, 2)
input_shape = X_train.shape[1:]

# Here is the actual neural network
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# NOTE: Changing # of layers, type of layers, and all inputs to the layers are fair game for optimization
# Conv Layer 1
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 2
model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Conv Layer 3
model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Conv Layer 3
model.add(Convolution2D(8, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Flatten and Dropout
model.add(Flatten())
model.add(Dropout(0.5))

# FC Layer 1
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC Layer 2
model.add(Dense(64))
model.add(Activation('relu'))

# FC Layer 3
model.add(Dense(32))
model.add(Activation('relu'))

# Final FC Layer - six outputs - the three coefficients for each of the two lane lines polynomials
model.add(Dense(6))

# Using a generator to help the model generalize/train better
datagen = ImageDataGenerator(rotation_range = 10, vertical_flip = True, height_shift_range = .1)
datagen.fit(X_train)

# Compiling and training the model
# Currently using MAE instead of MSE as MSE tends to only have 1 label for left curve, 1 for right curve, and 1 for straight (nothing in between)
model.compile(optimizer='Adam', loss='mean_absolute_error')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch = 5*len(X_train), nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# Show summary of model
model.summary()
