# MLND-Capstone
My capstone project for Udacity's Machine Learning Nanodegree

#### This project is in progress. I will be updating this repository with steps I have taken so far as well as current status and future updates needed.

Please see my original capstone proposal [here](https://github.com/mvirgo/MLND-Capstone-Proposal).

## Completed Steps
* Obtaining driving video
* Extracting images from video frames (see `load_videos.py`)
* Manual processing of images to remove unclear / blurry images
* Obtaining camera calibration for the camera used to obtain driving video (see `cam_calib.py`)
* Load in all road images (accounting for time series), and save to a pickle file (see `load_road_images.py`)
* Undistort images (using the camera calibration) and perspective transform to top-down images, save the transformed images (see `undistort_and_transform.py`)
* Created a file (based on my previous Advanced Lane Lines model, see `make_labels.py`) to calculate the lines, and to save the line data (which will be the labels for each image) to a pickle file. This file needs the lines re-drawn over the perspective transformed images in red to work appropriately.
* Built a neural network (see `unopt_NN.py`) that can take perspect transformed images and labels, then train to predict labels on new perspective transformed images. This neural network is *unoptimized* but can at least output the necessary six labels. I will continue to improve the model in later steps.
* Created a file (see `check_labels.py`) in which I save down each image after labelling, in order to check whether the labels appear correct given the re-drawn lane lines from a computer vision-based model. This will help make sure I'm feeding good labels to my neural network for training.
* Manually re-drew lane lines for detection in the `make_labels.py` file. Doing so involved re-drawing in red over the lane lines to aid the computer vision-based model to calculate the line data (especially helpful where the line is fairly unclear in the video image). I originally obtained nearly 700 seconds of video (over 11 minutes), which was over 21,000 frames. After manually getting rid of blurry images and others (such as those with little to no visible lines within the area for which the perspective transformation would occur), I had over 14,000 images. In order to account for similar images within small spans of time, I am currently using only 1 in 10 images, or 3 frames out of each second of video. As such, just over 1,400 images were used.

## Current Status
I am currently working with the images produced from the `check_labels.py` file to determine which images need further manual work on the redrawn line in order to have an accurate enough label for testing. The vast majority of images from straight or only slight curves do not have additional labelling issues, but extreme curves so far do not produce good labels even after manually re-drawing the lines to assist the computer vision-based model. I am just starting this part of the process but will check each of the 1,420 images with their lines redrawn to see whether they need improvement.

## Issues / Challenges so far
#### General
* File ordering - using `glob.glob` does not pull in images in a natural counting fashion, wherein 10 follows 9, but looks at the first digit followed by the second digit. I needed to add in an additional function (see Lines 13-16 in `load_road_images.py`) to get it to pull the images in normally. This is crucial to make sure the labelling is matched up with the same image later on.

#### Images
The below issues often caused me to have to throw out the image:
* Image blurriness - although road bumpiness is the main driver of this, it is pronounced in raining or nighttime conditions. The camera may focus on the rain on the windshield or on reflections from within the car.
* Line "jumping" - driving on bumpy roads at highway speeds tends to cause the lane lines to "jump" for an image or two. I deleted many of these although tried to keep some of the better ones to help make a more robust model.
* Dirt or leaves blocking lines
* Lines blocked by a car
* Intersections (i.e. no lane markings) and the openings for left turn lanes
* Extreme curves - lane line may be off to the side of the image
* Time series - especially when going slower, frame to frame images have little change and could allow the final model to "peek" into the validation data
* Lines not extending far enough down - although the lane lines may be visible in the regular image, they may disappear in the perspective-transformed image, making it impossible to label
* Given that I am manually drawing lines in red (for putting through the CV-based model for labelling of line data purposes), tail lights at night could potentially add unwanted data points in the thresholded images. Additionally, blue LED lights at night could also cause problems if I were to draw lines in blue. I have not as of yet looked at how much green is in each image, but assume grass or leaves could also cause issues.
* Certain images failed with the histogram and had to be slightly re-drawn, in a select few cases meaning the drawn line needed to be extended further than the original image showed. Isolating those with issues was made easier by including a counter in the middle of the file to make labels (not in the finished product) which identified which image failed the histogram test
* The CV-based model can still fail at creating good labels, leading to large differences that have an out-sized effect on the neural network training (especially when using mean squared error compared to other loss types). Prior to finalizing the model I will use the `check_labels.py` file to go back and either fix the images or remove them so that training can be improved.

## Upcoming
* Finish checking the accuracy of each label and fixing as necessary
* Creation of a second deep neural network to predict lane lines using:
  * a model that calculates the line prior to perspective transformation - perhaps using a keras crop layer to help focus the neural network's training on the important area of the images (i.e. below the horizon line). This model would *potentially* skip the need to ever perspective transform the original image.
* Optimization of the above model(s) (parameters, architecture, adding a python generator)
* Based on which neural network model I choose above, create a file to re-draw the lane lines
* Compare the original CV-based lane line model's loss with the neural network's (based on the improved labels from the manual drawn lines)
* Additionally, compare the speed of the original CV-based lane line model vs. the neural network
* Assess the performance of the neural network on additional videos (such as Challenge videos in the Udacity Advanced Lane Lines project)
* Complete final project write-up

#### Minor potential improvements
* The function `natural_key` is currently contained in both `load_road_images.py` and `make_labels.py`. This should be consolidated down (probably in a separate file; may also consolidate other helper functions within there).
* The `make_labels.py` file is currently a little inefficient from a memory perspective, as it pulls from one list into another. I should combine some of the functions to skip the middle list and avoid using extra memory.
