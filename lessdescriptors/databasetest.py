import cv2 as cv
import os
from glob import glob
import numpy as np
import time
import random

class Painting:
  def __init__(self, name, room, descriptors):
    self.name = name
    self.room = room
    self.descriptors = descriptors

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

sift = cv.SIFT_create()

def get_SIFTkeypoints(image):
	# find the keypoints and descriptors with SIFT
	# sift werkte niet op graywaarde (resize_image_to_width kreeg variabele image en niet variabele gray) en 400 -> 200
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	gray = resize_image_to_width(gray, 300)
	kp, des = sift.detectAndCompute(gray,None)

	return (kp,des)


bf = cv.BFMatcher()

def matches_ratiotest(matches):
    # Apply ratio test
    if(len(matches) <= 1):
        return 0
    good = []
    for m,n in matches:
        if m.distance < 0.60*n.distance:
            good.append([m])
    return len(good)

def compare_descriptors(descriptors_base, descriptors_compare):
    # BFMatcher with default params
	if descriptors_base is None or descriptors_compare is None or descriptors_base.size == 0 or descriptors_compare.size == 0:
		return 0
	
	matches = bf.knnMatch(descriptors_base,descriptors_compare,k=2)
	
	keypoints = matches_ratiotest(matches)
	return keypoints

def contour(image, kernel=(5,5), sigma=1.2, thresh_1=37, thresh_2=150, thresh_3=37, thresh_4=150, dilatekernel = (7,7), verschiltresh=110, sobbelkernel=3):
	# Zoekt contours van schilderijen en returnt deze als een image
	# Resizen is een optie

	# Resize + convert to grayscale
	#image = cv2.resize(src=image, dsize=(0, 0), dst=None, fx=0.5, fy=0.5)

	gray = cv.cvtColor(src=image, code=cv.COLOR_BGR2GRAY)

	gray = cv.GaussianBlur(src = gray, ksize = kernel, sigmaX = sigma)

	# Canny edge detector
	thr_1 = thresh_1
	thr_2 = thresh_2
	edges = cv.Canny(image=gray, threshold1=thr_1, threshold2=thr_2)
	#cv.imshow("edges", edges)


	# Dilatie
	dilated = cv.dilate(src=edges, kernel=np.ones(dilatekernel))
	#cv.imshow("dilated", dilated)

	# Contours vinden
	contours, hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	

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
		x, y, w, h = cv.boundingRect(c)
		if (h<0.99*imgh) and (w<0.99*imgw) and (h>0.2*imgh) and (w>0.2*imgw):
			tempcontours.append(c)
	
	if len(tempcontours) == 0:
		#print('Geen box')
		return []

	contours = tempcontours

	for contour in contours:
		polygon = cv.approxPolyDP(contour, 0.1 * cv.arcLength(contour, True), True)  # TODO, goede waarde vinden
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
		transform = cv.getPerspectiveTransform(approx_box, bounding_box)
		result = cv.warpPerspective(image, transform, (0, 0))

		
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


paintings = []

url = sorted(glob("/home/maartenvandenbrande/Documents/unif/computervisie/dataset_pictures_msk/*"))
urldescriptors = "//home//maartenvandenbrande//Documents//unif//computervisie//tests//lessdescriptors300_2//"




"""
array1 = np.load("database_factor0.5.npy", allow_pickle=True)
array2 = np.load("database_factor0.77.npy", allow_pickle=True)

som1 = 0
som2 = 0
print(array1[0][2].shape)
print(len(array1[0][2]))

for element in array1:
	som1 += len(element[2])

for element in array2:
	som2 += len(element[2])

print("len array1: %s --- len array2: %s --- percent: %s" % (som1, som2, som2/som1*100))
"""






for database_nummer in range(20,250,10):
	juist = 0
	totaal = 0	
	database_start_time = time.time()
	array = np.load(urldescriptors + "database_descriptors" + str(database_nummer) + ".npy", allow_pickle=True)
	print("database nummer: %s" % database_nummer)
	database_avg_match_time = (0,0) #(number of maches, total match time)
	for element in array:
		p = Painting(element[0], element[1], element[2])
		paintings.append(p)
	progress = 0
	database_contours_found = 0
	for zaal in url:
		print("progress: %0.2f percent" % (progress/len(url)*100), end='\r')
		for image_url in sorted(glob(zaal+"/*")):
			#t0 = time.time()
			base=os.path.basename(image_url) 
			image_name = os.path.splitext(base)[0]
			image_name = image_name.replace("IMG_", "")
			image_name_splitted = image_name.split('_')
			name = image_name_splitted[1]
			image = cv.imread(image_url)
			
			#t1 = time.time()
			image = resize_image_to_width(image, 300)
			#cv.imshow("test", image)
			contours = contour(image)
			#print("number: %s" % (len(contours)))
			#cv.waitKey(0)
			#exit()

			if(len(contours)>0):
				database_contours_found += 1

			#t2 = time.time()
			found_image_names = []
			for contour_image in contours:
				start_time_contour = time.time()
				pkeys, pdesc = get_SIFTkeypoints(contour_image)

				list_name_feature_scores = []
				max_keypoints = 0

				for image in paintings:
						keypoints = compare_descriptors(image.descriptors, pdesc)

						list_name_feature_scores.append((image.name, image.room, keypoints))

						if keypoints > max_keypoints:
							max_keypoints = keypoints

				if max_keypoints == 0:
					continue

				#list with total scores of each image 
				list_name_totalscore = []

				for name_score in list_name_feature_scores:
					calculated_score = name_score[2]/max_keypoints
					#?                           naam            zaal            totale score
					list_name_totalscore.append((name_score[0], name_score[1], calculated_score))

				found_image_names.append(sorted(list_name_totalscore, key = lambda x: x[2], reverse = True)[0][0])
				database_avg_match_time = (database_avg_match_time[0]+1, database_avg_match_time[1] + (time.time()-start_time_contour))

			#t3 = time.time()
			for found_name in found_image_names:
				found_name = found_name.split("-")[0]
				#print("name database: %s --- name image: %s" % (found_name, name))
				if(found_name == name):
					#print(True)
					juist += 1
					break
			
			#print("time --- init: %0.5f --- contour: %0.5f --- match: %0.5f --- end: %0.5f" % (t1-t0,t2-t1,t3-t2,time.time()-t3))

			totaal += 1
		progress += 1
	print(database_avg_match_time[1]/database_avg_match_time[0])
	print("painting found: %0.3f percent -- percent matched: %0.3f percent -- in %0.2f seconden with avg match time of: %0.5f" % (database_contours_found/totaal*100, juist/totaal*100, time.time()-database_start_time, database_avg_match_time[1]/database_avg_match_time[0]))



"""
array = np.load("database_50.npy", allow_pickle=True)

for element in array:
	p = Painting(element[0], element[1], element[2])
	paintings.append(p)

for zaal in url:
	for image_url in sorted(glob(zaal+"/*")):
		base=os.path.basename(image_url) 
		image_name = os.path.splitext(base)[0]
		image = cv.imread(image_url)

		t1 = time.time()
		pkeys, pdesc = get_SIFTkeypoints(image)

		list_name_feature_scores = []
		max_keypoints = 0

		for image in paintings:
				keypoints = compare_descriptors(image.descriptors, pdesc)

				list_name_feature_scores.append((image.name, image.room, keypoints))

				if keypoints > max_keypoints:
					max_keypoints = keypoints

		if max_keypoints == 0:
			break

		#list with total scores of each image 
		list_name_totalscore = []

		for name_score in list_name_feature_scores:
			calculated_score = name_score[2]/max_keypoints
			#?                           naam            zaal            totale score
			list_name_totalscore.append((name_score[0], name_score[1], calculated_score))

		list_name_totalscore = sorted(list_name_totalscore, key = lambda x: x[2], reverse = True)[:3]

		t2 = time.time()
		diff = t2 - t1
		print("total time: %s seconds" % (diff))

		naam_image_database = list_name_totalscore[0][0].split("_")[4]
		image_name = image_name.split("_")[1]

		print("name database: %s --- name image: %s" % (naam_image_database, image_name))
		if(naam_image_database == image_name):
			juist += 1

		totaal += 1



	print("totale score: %s" % juist/totaal*100)
"""