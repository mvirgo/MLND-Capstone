"""Pull in road images - currently uses 1 in 10 images
in order to account for time series data (i.e. very similar
images over a short period of time due to 30 fps video.
Also saves a pickle file for later use.
"""

import os
import glob
import matplotlib.image as mpimg
import pickle
import re

# Load road image locations in order
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def pull_images():
    """ Note that I had 12 videos for which I extracted the
    image frames into separate folders. For folders in that
    range, I pull in the images in a sorted order - SUPER
    IMPORTANT to keep them in order for later labelling. To
    account for similar images in short time spans (since video
    is 30 frames per second), one in ten images, or three per
    second, are going to be used). These are appended to a list.
    """
    for fold in range(0,13):
        road_image_locs = glob.glob('my_vid/%d/*.jpg' % fold)
        sort_road_image_locs = sorted(road_image_locs, key=natural_key)
        counter = 0
        for fname in sort_road_image_locs:
            counter += 1
            if counter % 10 == 0:
                img = mpimg.imread(fname)
                road_images.append(img)     

# List for images
road_images = []

# Pull in the desired images       
pull_images()

# Save the images to a pickle file
pickle.dump(road_images,open('road_images.p', "wb" ))
