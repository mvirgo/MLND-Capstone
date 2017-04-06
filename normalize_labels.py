""" Example of normalization technique to be included in training
the neural network. The means and standard deviations for each label
will eventually be saved in order to be used in reversing
normalization after neural network prediction.
"""
import numpy as np
import pickle

def norm_inputs(labels):
    """ Appends each of the six line coefficients to a list.
    Then, calculates the mean and standard deviation of each coefficient.
    """
    labels_by_coeff = [[],[],[],[],[],[]]
    means = []
    std_devs = []

    for label in labels:
        for n in range(len(label)):
            labels_by_coeff[n].append(label[n])

    for coeff_list in labels_by_coeff:
        means.append(np.mean(coeff_list))
        std_devs.append(np.std(coeff_list))
        
    return means, std_devs

def norm_label(label):
    """ Normalizes each coefficient within a given label using
    the respective means and standard deviations.
    """
    for n in range(len(label)):
        label[n] = (label[n] - means[n]) / std_devs[n]
    return label

def rev_norm_label(label):
    """ Reverses normalization, for use after the neural network
    has output its prediction.
    """
    for n in range(len(label)):
        label[n] = label[n] * std_devs[n] + means[n]
    return label

# Load labels
labels = pickle.load(open( "lane_labels.p", "rb" ))

# Calculate the means and standard deviations of the labels
means, std_devs = norm_inputs(labels) 

# Normalize all labels - these are then ready to be split for training/validation
for n, label in enumerate(labels):
    labels[n] = norm_label(label)

