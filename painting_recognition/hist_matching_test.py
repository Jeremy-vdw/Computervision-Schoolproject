import cv2 as cv
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def get_histogram(image):

    hist = cv.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()

    return hist 

def get_NxN_histograms(image, N=5):

    block_height = int(image.shape[0]/N)
    block_width = int(image.shape[1]/N)

    histograms = np.empty([N, N], dtype=object)

    for row in range(0, image.shape[0] - block_height + 1, block_height):
        for col in range(0, image.shape[1] - block_width + 1, block_width):
            block = image[row:row + block_height, col:col + block_width]

            row_num = int(row/block_height)
            col_num = int(col/block_width)
            histograms[row_num][col_num] = get_histogram(block)

    return histograms

#https://stackoverflow.com/questions/15007304/histogram-equalization-not-working-on-color-image-opencv
def equalize_rgbcolor(image):
    ycrcb=cv.cvtColor(image,cv.COLOR_BGR2YCR_CB)
    channels=cv.split(ycrcb)
    cv.equalizeHist(channels[0],channels[0])
    cv.merge(channels,ycrcb)
    cv.cvtColor(ycrcb,cv.COLOR_YCR_CB2BGR,image)
    return image

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

def comparison_NxN(hist1, hist2, method):
    
    block_histogram_score = 0
    for row in range(0, len(hist1)):
        for col in range(0, len(hist1[row])):
            block_histogram_score += cv.compareHist(hist1[row][col], hist2[row][col], method)
    block_histogram_score /= len(hist1)*len(hist1)

    return block_histogram_score
        

folder = sorted(glob("/Users/jeremyvandewalle/Desktop/test/*"))
folder = sorted(glob("/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Database/*"))

index = {}
images = {}

for file in folder:

    image = cv.imread(file)
    image = resize_image_to_width(image, width=600)
    image = equalize_rgbcolor(image)
    images[file] = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    histograms = get_NxN_histograms(image)

    index[file] = histograms


OPENCV_METHODS = (("Correlation", cv.HISTCMP_CORREL),
	("Chi-Squared", cv.HISTCMP_CHISQR),
	("Intersection", cv.HISTCMP_INTERSECT),
	("Hellinger", cv.HISTCMP_BHATTACHARYYA))

for (methodName, method) in OPENCV_METHODS:
	# initialize the results dictionary and the sort
	# direction
	results = {}
	reverse = False
	# if we are using the correlation or intersection
	# method, then sort the results in reverse order
	if methodName in ("Correlation", "Intersection"):
		reverse = True
	
	time1 = time.time()
	# loop over the index
	for (k, hist) in index.items():
		# compute the distance between the two histograms
		# using the method and update the results dictionary
		d = comparison_NxN(index["/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Database/painting14.png"], hist, method)
		score = d
		results[k] = score
	# sort the results
	results = sorted([(v, k) for (k, v) in results.items()], reverse = reverse)[:8]
	time2 = time.time()
	diff = time2 - time1
	print("Method: " + methodName + " : " + str(diff))

    # show the query image
	fig = plt.figure("Query")
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(images["/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Database/painting14.png"])
	plt.axis("off")
	# initialize the results figure
	fig = plt.figure("Results: %s" % (methodName))
	fig.suptitle(methodName, fontsize = 20)
	# loop over the results
	for (i, (v, k)) in enumerate(results):
		# show the result
		ax = fig.add_subplot(1, len(results), i + 1)
		ax.set_title("%.2f" % (v))
		plt.imshow(images[k])
		'''
		color = ('b','g','r')
		for i,col in enumerate(color):
			histr = cv.calcHist(images[k],[i],None,[16],[0,256])
			plt.plot(histr,color = col)
			plt.xlim([0,16])
		'''
		plt.axis("off")
# show the OpenCV methods
plt.show()
