import cv2 as cv 
import numpy as np
import time
from glob import glob
# TODO: move image functions

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

def cut_image(image, points):
		box = points.reshape(4, 2)
		approx_box = np.zeros((4, 2), dtype="float32")
		
		# Caclulate sum
		sum = box.sum(axis=1)
		approx_box[0] = box[np.argmin(sum)]
		approx_box[2] = box[np.argmax(sum)]
	
		# Calculate difference
		diff = np.diff(box, axis=1)
		approx_box[1] = box[np.argmin(diff)]
		approx_box[3] = box[np.argmax(diff)]

		# Determine width and height of bounding box
		smallest_x = 1000000
		smallest_y = 1000000
		largest_x = -1
		largest_y = -1

		for point in approx_box:
			if point[0] < smallest_x:
				smallest_x = point[0]
			if point[0] > largest_x:
				largest_x = point[0]
			if point[1] < smallest_y:
				smallest_y = point[1]
			if point[1] > largest_y:
				largest_y = point[1]
	
		maxWidth = int(largest_x - smallest_x)
		maxHeight = int(largest_y - smallest_y)

		bounding_box = np.array([
			[0, 0],
			[maxWidth, 0],
			[maxWidth, maxHeight],
			[0, maxHeight]], dtype="float32")

		# Apply transformation
		transform = cv.getPerspectiveTransform(approx_box, bounding_box)
		result = cv.warpPerspective(image, transform, (0, 0))
		# Crop out of original picture
		extracted = result[0:maxHeight, 0:maxWidth]
		return extracted

## 1 Mean Shift Segmentation
def mean_shift_segmentation(image):
    return cv.pyrMeanShiftFiltering(image, 13, 17, maxLevel=4)

## 2 Create a Mask of the Largest Segment
def mask_largest_segment(image, step):
    
    mask = np.zeros((image.shape[0]+2,image.shape[1]+2),np.uint8)

    floodflags = 4
    floodflags |= cv.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    largest_mask = None
    current_size = 0
    largest_segment_size = 0

    for y in range(0, image.shape[0], step): #height
        for x in range(0, image.shape[1], step): #width
            seed = (x,y)
            num,im,mask,rect = cv.floodFill(image, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
            x, y, w, h = rect
            current_size = w * h 
            if current_size > largest_segment_size:
                largest_segment_size = current_size
                largest_mask = mask
    
    return largest_mask

def auto_canny(img, s=0.33):
    v = np.median(img)
    
    l = int(max(0, (1.0 - s)*v))
    u = int(min(255, (1.0 + s) * v))
    edged = cv.Canny(img, l, u)
    return edged

## 3 Invert, add Erosion image 
def erosion(mask):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    mask = cv.bitwise_not(mask)
    mask = cv.erode(mask, kernel)
    mask = cv.medianBlur(mask, 11)
    edges = auto_canny(mask)
    return edges


## 5 Connected Components Analysis --> Just for visualisation
def connectedComponents(mask):
    num_labels, labels = cv.connectedComponents(mask)

    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

### 4 Find contours: 
def find_contours(mask, image = None):
    contours_painting = []
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        epsilon = 0.1*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)
        if len(approx) == 4:
            area_percentage = cv.contourArea(approx) / (mask.shape[0] * mask.shape[1])
            if area_percentage > 0.01:
                contours_painting.append(approx)
                pts = approx.reshape((-1,1,2))
                #cv.polylines(image,[pts],True,(255,0,255))
                area = cv.contourArea(approx)
    return contours_painting

'''
imagepath_folder = sorted(glob("/Users/jeremyvandewalle/Desktop/20190217_110709.jpg"))

def detect_painting(image):
    image = resize_image_to_width(image, 400)
    image_copy = image
    image = mean_shift_segmentation(image)
    mask = mask_largest_segment(image, 50)
    mask = erosion(mask)
    cv.imshow("image", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    contours = find_contours(mask, image_copy)
    
    
    paintings = []
    for c in contours:
        painting = cut_image(image_copy, c)
        paintings.append(painting)
        cv.imshow("image", painting)
        cv.waitKey(0)
        cv.destroyAllWindows()

for path in imagepath_folder:
    image = cv.imread(path)
    detect_painting(image)
'''

def detect_contours(image):
    time1 = time.time()
    image = mean_shift_segmentation(image)
    mask = mask_largest_segment(image, 50)
    mask = erosion(mask, image)
    contours = find_contours(mask)
    time2 = time.time()
    #print("found " + str(len(contours)) + " paintings in " + str(time2 - time1))
    return contours



