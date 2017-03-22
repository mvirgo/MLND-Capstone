"""Code obtained from http://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
This file will extract individual images from video frames
Note that the file type of the videos and image can be changed below
Tested with .MOV or .mp4 for movies and either .jpg or .png for images
"""

import cv2

# Enter the location of the video - add folders if not in current directory
video_location = 'Videos/1.MOV'

vidcap = cv2.VideoCapture(video_location)
success, image = vidcap.read()
count = 0
success = True

# Iterates through all video frames until it runs out (i.e. video ends)
# Change for desired location to save image files extracted
# If putting in a folder, folder must have already been created
while success:
    success, image = vidcap.read()
    cv2.imwrite('my_vid/frame%d.jpg' % count, image)
    count += 1
