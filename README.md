# MLND-Capstone
My capstone project for Udacity's Machine Learning Nanodegree

### Lane Detection with Deep Learning
In this project, I use a deep learning-based approach to improve upon lane detection. My final model uses a fully convolutional neural network to output an image of a predicted lane.

Please see my final Capstone Project Report [here](MLND%20Capstone%20Project%20Report.pdf).

Also, see my original capstone proposal [here](proposal.pdf).

Lastly, check out the wiki page [in this repository](https://github.com/mvirgo/MLND-Capstone/wiki) to see some more of my steps along the way. The separate ["early_steps" branch](https://github.com/mvirgo/MLND-Capstone/tree/early_steps) contains earlier code for previous versions of the neural network as well as files that can extract data for training and perform some automatic labeling.

See an early version of the model detecting lane lines with perspective transformed images [here.](https://youtu.be/ZZAgcSqAU0I)
An early version of my model trained *without* perspective transformed images, i.e. regular road images, can be seen [here!](https://www.youtube.com/watch?v=Vq0vlKdyXnI)

Lastly, with the finalized fully convolutional model, there are a couple additional videos I made. The first, which is the same video from the above two, has between 10-20% of the frames fed into the mode, as can be seen [here.](https://youtu.be/bTMwF1UoZ68) Additionally, a video made from the Challenge Video from Udacity's Advanced Lane Lines project in the SDCND, where the neural network had **never** seen the video before, can be seen [here.](https://youtu.be/_qwET69bYa8) The model performs fairly robustly on the never-before-seen video, with the only hitch due to the large light difference as it goes under the overpass.

An additional video can be seen at [this Dropbox link.](https://www.dropbox.com/s/18jia2x9pg42s4n/proj_reg_vid.mp4?dl=0)

## Dataset
#### For fully convolutional network
You can download the full training set of images I used [here](https://www.dropbox.com/s/rrh8lrdclzlnxzv/full_CNN_train.p?dl=0) and the full set of 'labels' (which are just the 'G' channel from an RGB image of a re-drawn lane with an extra dimension added to make use in Keras easier) [here](https://www.dropbox.com/s/ak850zqqfy6ily0/full_CNN_labels.p?dl=0) (157 MB).

#### Images with coefficient labels
If you just want the original training images with no flips or rotations (downsized to 80x160x3) you can find them [here](https://www.dropbox.com/s/1bnp70bhaz5kma9/coeffs_train.p?dl=0). You can also find the related coefficient labels (i.e. not the drawn lane labels, but the cofficients for a polynomial line) [here](https://www.dropbox.com/s/ieulvrcooetrlmd/coeffs_labels.p?dl=0).

## Software Requirements
You can use [this conda environment file](lane_environment.yml). In the command line, use `conda env create -f lane_environment.yml` and then `source activate lane_environment` (or just `activate` with the environment name on Windows) to use the environment.

## Key Files
Although I have included many of the python files I created to help process my images and various prototype neural networks in the ["early_steps" branch](https://github.com/mvirgo/MLND-Capstone/tree/early_steps), the key files are:
* [fully_conv_NN.py](fully_conv_NN.py) - Assuming you have downloaded the training images and labels above, this is the fully convolutional neural network to train using that data.
* [full_CNN_model.h5](full_CNN_model.h5) - These are the final outputs from the above CNN. Note that if you train the file above the originals here will be overwritten! These get fed into the below.
* [draw_detected_lanes.py](draw_detected_lanes.py) - Using the trained model and an input video, this predicts the lane, averages across 5 frames, and returns the original video with predicted lane lines drawn onto it. Note that it is currently set up to use the basic video from Udacity's SDCND Advanced Lane Lines project [here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4), but the code at the end can be changed to accept different input videos.

## Training Image Statistics
* 21,054 total images gathered from 12 videos (a mix of different times of day, weather, traffic, and road curvatures)
* 17.4% were clear night driving, 16.4% were rainy morning driving, and 66.2% were cloudy afternoon driving
* 26.5% were straight or mostly straight roads, 30.2% were a mix or moderate curves, and 43.3% were very curvy roads
* The roads also contain difficult areas such as construction and intersections
* 14,235 of the total that were usable of those gathered (mainly due to blurriness, hidden lines, etc.)
* 1,420 total images originally extracted from those to account for time series (1 in every 10)
* 227 of the 1,420 unusable due to the limits of the CV-based model used to label (down from 446 due to various improvements made to the original model) for a total of 1,193 images
* Another 568 images (of 1,636 pulled in) gathered from more curvy lines to assist in gaining a wider distribution of labels (1 in every 5 from the more curved-lane videos; from 8,187 frames)
* In total, 1,761 original images
* I pulled in the easier project video from Udacity's Advanced Lane Lines project (to help the model learn an additional camera's distortion) - of 1,252 frames, I used 1 in 5 for 250 total, 217 of which were usable for training
* A total of 1,978 actual images used between my collections and the one Udacity video
* After checking histograms for each coefficient of each label for distribution, I created an additional 4,404 images using small rotations of the images outside the very center of the original distribution of images. This was done in three rounds of slowly moving outward from the center of the data (so those further out from the center of the distribution were done multiple times). 6,382 images existed at this point.
* Finally, I added horizontal flips of each and every road image and its corresponding label, which doubled the total images. All in all, there were a total of 12,764 images for training.
