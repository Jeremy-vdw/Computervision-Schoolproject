import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

def equalize_rgbcolor(image):
    ycrcb=cv.cvtColor(image,cv.COLOR_BGR2YCR_CB)
    channels=cv.split(ycrcb)
    cv.equalizeHist(channels[0],channels[0])
    cv.merge(channels,ycrcb)
    cv.cvtColor(ycrcb,cv.COLOR_YCR_CB2BGR,image)
    return image

# Initiate SIFT detector
sift = cv.SIFT_create()

def get_SIFTkeypoints(image):
    # find the keypoints and descriptors with SIFT
	 # sift werkte niet op graywaarde (resize_image_to_width kreeg variabele image en niet variabele gray) en 400 -> 200
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = resize_image_to_width(gray, 300)
    kp, des = sift.detectAndCompute(gray,None)
	 
    return (kp,des)

def normalize_image(image):
     image = resize_image_to_width(image, width=600)
     image = equalize_rgbcolor(image)
     return image

def get_histogram(image):

     hist = cv.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
     hist = cv.normalize(hist, hist).flatten()

     return hist 

def get_part_histograms(image):

     image = normalize_image(image)

     height_part = int(image.shape[0]/5)
     width_part = int(image.shape[1]/5)

     histograms = np.empty([5, 5], dtype=object)

     for y in range(0, image.shape[0] - height_part + 1, height_part):
         for x in range(0, image.shape[1] - width_part + 1, width_part):
             part = image[y:y + height_part, x:x + width_part]

             r = int(y/height_part)
             c = int(x/width_part)
             histograms[r][c] = get_histogram(part)

     return histograms