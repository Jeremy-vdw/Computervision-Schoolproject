from glob import glob
from re import split
import cv2
from matplotlib import pyplot as plt
import time


def loadImage(imagePath):
	"""
	Loads and returns an image
	Parameters
	----------
		imagePath : string
			The absolute or relative path pointing to an image.
	Returns
	-------
		The image
	"""
	return cv2.imread(imagePath)


def createWindowAtCoordinates(windowname, x, y):
	"""
	Creates a window at the specific location
	"""
	cv2.namedWindow(windowname)
	cv2.moveWindow(windowname, x, y)


def showImage(windowname, image, delay=0):
	"""
	Displays an image in the specified window
	"""

	cv2.imshow(windowname, image)
	cv2.waitKey(delay)


def drawLines(image, lines):
	"""
	Draws lines which are the result of the HoughLinesP function onto the image.
	"""
	if lines is not None:
		for line in lines:
			line = line[0]  # a line is a 2D array for compatibility with C++, but will only contain one row, which contains 4 numerical values

			cv2.line(img=image,
					 pt1=(line[0], line[1]), pt2=(line[2], line[3]),
					 color=(0,255,0), thickness=5, lineType=cv2.LINE_AA)


def drawPoints(image, points, color):
	"""
	Draws circles around intersections.
	"""
	for point in points:
		x = point[0]
		y = point[1]
		cv2.circle(img=image, center=(int(x), int(y)), radius=5,
				   color=color, thickness=5)


def showImagesHorizontally(windowname, delay=None, *images):
	"""
	Shows multiple images horizontally in the specified window
	"""
	showImage(windowname, cv2.hconcat((images)), delay)


def saveImage(image, savePath):
	"""
	Saves an image to the specified file.
	"""
	cv2.imwrite(savePath, image)


def resizeImage(image, dimension):
	"""
	Resizes the image to the given dimension
	Parameters
	----------
		image : numpy.ndarray
			The image to resize
		dimension : tuple
			A tuple (width, height) representing the dimension to resize to
	Returns
	-------
			The resized image
	"""
	return cv2.resize(src=image, dsize=dimension)

def showPlot(histogram):
	plt.plot(histogram)
	plt.xlim([0,256])
	plt.show()

def compareHistImage(img1, img2, histogram1, histogram2):
	plt.subplot(221), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
	plt.subplot(222), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
	plt.subplot(223), plt.plot(histogram1), plt.xlim([0,256])
	plt.subplot(224), plt.plot(histogram2), plt.xlim([0,256])
	plt.show()

def compareHistGrayImage(img1, img2, histogram1, histogram2):
	plt.subplot(221), plt.imshow(img1)
	plt.subplot(222), plt.imshow(img2)
	plt.subplot(223), plt.plot(histogram1), plt.xlim([0,256])
	plt.subplot(224), plt.plot(histogram2), plt.xlim([0,256])
	plt.show()

def resize_image_to_width(image, width):
	height = int(width / image.shape[1] * image.shape[0])
	dimensions = (width, height)
	resized = cv2.resize(image, dimensions, interpolation=cv2.INTER_NEAREST)
	
	return resized

grondplanurl = glob("floorplans/*")
grondplan_rooms = {}
for grondplan in grondplanurl:
	image = cv2.imread(grondplan)
	image = resize_image_to_width(image, 300)
	grondplan_rooms[grondplan.split("/")[-1].split(".")[0]] = image

grondplan_height , grondplan_width, RGB = image.shape

def show_room(rooms, frame):
	try:
		resized_frame = resize_image_to_width(frame, 1500)
		resized_frame[0:grondplan_height,0:grondplan_width,:] = grondplan_rooms.setdefault(rooms[0], grondplan_rooms["default"])
		cv2.imshow("Project computervisie: Video", resized_frame)
	except:
		print("failed to show image")