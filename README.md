# MLND-Capstone
My capstone project for Udacity's Machine Learning Nanodegree

#### This project is in progress. I will be updating this repository with steps I have taken so far as well as current status and future updates needed.

Please see my original capstone proposal [here](https://github.com/mvirgo/MLND-Capstone-Proposal).

## Completed Steps
* Obtaining driving video
* Extracting images from video frames (see `1_load_videos.py`)
* Manual processing of images to remove unclear / blurry images
* Obtaining camera calibration for the camera used to obtain driving video (see `2_cam_calib.py`)
* Load in all road images (accounting for time series), and save to a pickle file (see `3_load_road_images.py`)
* Undistort images (using the camera calibration) and perspective transform to top-down images, save the transformed images (see `4_undistort_and_transform.py`)

## Current Status
I am currently in the process of manually drawing over the lane lines in the perspective transform images. Doing so helps to supplement the detection of the lines, as I was use computer vision techniques in order to calculate the line data needed for labels for the eventual training of my neural network.

I originally obtained nearly 700 seconds of video (over 11 minutes), which was over 21,000 frames. After manually getting rid of blurry images and others (such as those with little to no visible lines within the area for which the perspective transformation would occur), I had over 14,000 images. In order to account for similar images within small spans of time, I am currently using only 1 in 10 images, or 3 frames out of each second of video. As such, I am going to end up manually re-drawing the lines on over 1,400 images. Due to some of the issues identified below, I have had to throw out a couple rounds of doing this manual re-draw, and am currently only at around 100 of the 1,400 images I will use for my initial neural network training.

## Issues / Challenges so far
#### General
* File ordering - using `glob.glob` does not pull in images in a natural counting fashion, wherein 10 follows 9, but looks at the first digit followed by the second digit. I needed to add in an additional function (see Lines 13-16 in `3_load_road_images.py`) to get it to pull the images in normally. This is crucial to make sure the labelling is matched up with the same image later on.

#### Images
* TO UPDATE

## Upcoming 
* TO UPDATE
