import numpy as np
import cv2
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import model_from_json
from scipy.misc import imresize
from numpy import newaxis

# Load undistortion information
mtx = pickle.load(open('mtx.p', "rb" ))
dist = pickle.load(open('dist.p', "rb" ))

# Load perspective transformation information
perspective_M = pickle.load(open('perspective_matrix.p', "rb" ))
Minv = np.linalg.inv(perspective_M)

# Load Keras model
json_file = open('model.json', 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights('model.h5')

# Load the label scaler to reverse the normalization
label_scaler = pickle.load(open( "scaler.p", "rb" ))

def birds_eye(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert to grayscale
    gray = cv2.cvtColor(undist,cv2.COLOR_RGB2GRAY)
    offset = 300 # offset for dst points
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])
    src = np.float32([[755,425],[1200, 720],[80,720],[525, 425]])
    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(gray, M, img_size)
    return warped, gray

def image_process(i,mtx,dist):
    img, other = birds_eye(np.copy(i),mtx,dist)
    img = imresize(img, (45, 80))
    img = img[..., newaxis]
    img = (img / 255) * .8 + 1
    img = np.array(img)
    img = img[None,:,:,:]
    return img, other

def road_lines(image):
    test_img, other = image_process(image,mtx,dist)

    prediction = model.predict(test_img)[0]
    prediction = label_scaler.inverse_transform(prediction)

    left_fit = prediction[0:3]
    right_fit = prediction[3:]

    fity = np.linspace(0, other.shape[0]-1, other.shape[0])
    fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
    fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]

    warp_zero = np.zeros_like(other).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

vid_output = 'reg_vid.mp4'

clip1 = VideoFileClip("Videos/1.MOV")

vid_clip = clip1.fl_image(road_lines)
vid_clip.write_videofile(vid_output, audio=False)
