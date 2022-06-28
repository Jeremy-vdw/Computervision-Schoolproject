import cv2
import numpy as np
import random
import time

def contour(image, kernel=(5,5), sigma=1.2, thresh_1=37, thresh_2=150, dilatekernel = (7,7), verschiltresh=110, sobbelkernel=3):
	# Zoekt contours van schilderijen en returnt deze als een image
	# Resizen is een optie

	# Resize + convert to grayscale
	#image = cv2.resize(src=image, dsize=(0, 0), dst=None, fx=0.5, fy=0.5)

	gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

	blurred = cv2.GaussianBlur(src = gray, ksize = kernel, sigmaX = sigma)

	# Canny edge detector
	thr_1 = thresh_1
	thr_2 = thresh_2
	edges = cv2.Canny(image=blurred, threshold1=thr_1, threshold2=thr_2)
	#cv2.imshow("edges", edges)

	# Dilatie
	dilated = cv2.dilate(src=edges, kernel=np.ones(dilatekernel))
	#cv2.imshow("dilated", dilated)

	# Contours vinden
	contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Sorteer countours op grootte. Grotere contours gaan schilderijen zijn
	#contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
	# https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html #4
	# Approximate the contours into a polygon (starting with the largest contour)
	# This will yield the first polygon with 4 points

	boxes = []

	imgh = image.shape[0]	
	imgw = image.shape[1]

	#soms neemt findContours het volledige beeld als contour dit haalt deze er uit, ook de vloer dat over de volledige breedte loopt wordt eruit gehaald
	tempcontours = []
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		if (h<0.99*imgh) and (w<0.99*imgw) and (h>0.1*imgh) and (w>0.1*imgw):
			tempcontours.append(c)
	
	if len(tempcontours) == 0:
		#print('Geen box')
		return []

	contours = tempcontours

	for contour in contours:
		polygon = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)  # TODO, goede waarde vinden
		if len(polygon) == 4:
			boxes.append(polygon)

	if len(boxes) == 0:
		#print('Geen box')
		return []

	returns = []
	for box in boxes:
		box = box.reshape(4, 2)
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
		transform = cv2.getPerspectiveTransform(approx_box, bounding_box)
		result = cv2.warpPerspective(image, transform, (0, 0))

		
		# Crop out of original picture
		extracted = result[0:maxHeight, 0:maxWidth]
		# extracted = gui.resizeImage(extracted, dimension=(500, 500))

		#haalt een groot deel van de false positives er uit, checked enkele keren het verschil tussen 2 pixels en indien kleine verschillen in kleurwaarden zal het geen schilderij zijn
		aantalpixels = 100
		verschil = 0
		for aantalkeer in range(0, aantalpixels):
			pixel1 = extracted[random.randrange(0, maxHeight, 1)][random.randrange(0, maxWidth, 1)]
			pixel2 = extracted[random.randrange(0, maxHeight, 1)][random.randrange(0, maxWidth, 1)]
			if(pixel1[0]>pixel2[0]):
				verschil += abs(pixel1[0]-pixel2[0])
			else:
				verschil += abs(pixel2[0]-pixel1[0])
			if(pixel1[1]>pixel2[1]):
				verschil += abs(pixel1[1]-pixel2[1])
			else:
				verschil += abs(pixel2[1]-pixel1[1])
			if(pixel1[2]>pixel2[2]):
				verschil += abs(pixel1[2]-pixel2[2])
			else:
				verschil += abs(pixel2[2]-pixel1[2])
		verschil = verschil/aantalpixels

		if verschil > verschiltresh:
			returns.append(extracted)
	return returns


def contour_pointers(image, kernel=(5,5), sigma=1.2, thresh_1=37, thresh_2=150, dilatekernel = (7,7), verschiltresh=110, sobbelkernel=3):
	# Zoekt contours van schilderijen en returnt deze als een image
	# Resizen is een optie

	# Resize + convert to grayscale
	#image = cv2.resize(src=image, dsize=(0, 0), dst=None, fx=0.5, fy=0.5)
	image = cv2.GaussianBlur(src = image, ksize = kernel, sigmaX = sigma)

	gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

	"""
	# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
	filtered_image = cv2.Laplacian(gray, ksize=sobbelkernel, ddepth=cv2.CV_16S)
	# converting back to uint8
	filtered_image = cv2.convertScaleAbs(filtered_image)
	cv2.imshow("filtered", filtered_image)
	"""

	# Canny edge detector
	thr_1 = thresh_1
	thr_2 = thresh_2
	edges = cv2.Canny(image=gray, threshold1=thr_1, threshold2=thr_2)

	"""
	#sobel edge detection
	sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=sobbelkernel)
	edges = cv2.convertScaleAbs(sobel)
	cv2.imshow("edges", edges)
	"""

	# Dilatie
	dilated = cv2.dilate(src=edges, kernel=np.ones(dilatekernel))
	#cv2.imshow("dilated", dilated)

	# Contours vinden
	contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	

	# Sorteer countours op grootte. Grotere contours gaan schilderijen zijn
	#contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
	boxes = []
	# https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html #4
	# Approximate the contours into a polygon (starting with the largest contour)
	# This will yield the first polygon with 4 points
	

	imgh = image.shape[0]	
	imgw = image.shape[1]

	#soms neemt findContours het volledige beeld als contour dit haalt deze er uit, ook de vloer dat over de volledige breedte loopt wordt eruit gehaald
	tempcontours = []
	for c in contours:
		x, y, w, h = cv2.boundingRect(c)
		if (h<0.99*imgh) and (w<0.99*imgw) and (h>0.05*imgh) and (w>0.05*imgw):
			tempcontours.append(c)
	
	if len(tempcontours) == 0:
		#print('Geen box')
		return []

	contours = tempcontours

	for contour in contours:
		polygon = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)  # TODO, goede waarde vinden
		if len(polygon) == 4:
			boxes.append(polygon)

	return boxes