""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Deconvolution2D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Load training images
train_images = pickle.load(open("full_CNN_train.p", "rb" ))

# Load image labels
labels = pickle.load(open("full_CNN_labels.p", "rb" ))

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)
# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 150
epochs = 20
pool_size = (2, 2)
input_shape = X_train.shape[1:]

### Here is the actual neural network ###
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
model.add(Convolution2D(60, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv1'))

# Conv Layer 2
model.add(Convolution2D(50, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv2'))

# Pooling 1
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 3
model.add(Convolution2D(40, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))

# Conv Layer 4
model.add(Convolution2D(30, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))

# Conv Layer 5
model.add(Convolution2D(20, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))

# Pooling 2
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 6
model.add(Convolution2D(10, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv6'))
model.add(Dropout(0.2))

# Conv Layer 7
model.add(Convolution2D(5, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv7'))
model.add(Dropout(0.2))

# Pooling 3
model.add(MaxPooling2D(pool_size=pool_size))

# Upsample 1
model.add(UpSampling2D(size=pool_size))

# Deconv 1
model.add(Deconvolution2D(10, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[8].output_shape, name = 'Deconv1'))
model.add(Dropout(0.2))

# Deconv 2
model.add(Deconvolution2D(20, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[7].output_shape, name = 'Deconv2'))
model.add(Dropout(0.2))

# Upsample 2
model.add(UpSampling2D(size=pool_size))

# Deconv 3
model.add(Deconvolution2D(30, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[5].output_shape, name = 'Deconv3'))
model.add(Dropout(0.2))

# Deconv 4
model.add(Deconvolution2D(40, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[4].output_shape, name = 'Deconv4'))
model.add(Dropout(0.2))

# Deconv 5
model.add(Deconvolution2D(50, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[3].output_shape, name = 'Deconv5'))
model.add(Dropout(0.2))

# Upsample 3
model.add(UpSampling2D(size=pool_size))

# Deconv 6
model.add(Deconvolution2D(60, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[1].output_shape, name = 'Deconv6'))

# Final layer - only including one channel so 1 filter
model.add(Deconvolution2D(1, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', 
                          output_shape = model.layers[0].output_shape, name = 'Final'))

### End of network ###


# Using a generator to help the model use less data
# I found NOT using any image augmentation here surprisingly yielded slightly better results
# Channel shifts help with shadows but overall detection is worse
datagen = ImageDataGenerator()
datagen.fit(X_train)

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), samples_per_epoch = len(X_train),
                    nb_epoch=epochs, verbose=1, validation_data=(X_val, y_val))

# Save model architecture and weights
model_json = model.to_json()
with open("full_CNN_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('full_CNN_model.h5')

# Show summary of model
model.summary()

