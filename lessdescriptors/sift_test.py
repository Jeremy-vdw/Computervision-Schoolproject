import cv2 as cv
import os
from glob import glob
import numpy as np
import time

from numpy.lib.utils import safe_eval

cv.namedWindow("lepel")

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

sift = cv.SIFT_create()

def get_SIFTkeypoints(image):
    # find the keypoints and descriptors with SIFT
	 # sift werkte niet op graywaarde (resize_image_to_width kreeg variabele image en niet variabele gray) en 400 -> 200
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = resize_image_to_width(gray, 200)
    kp, des = sift.detectAndCompute(gray,None)
	 
    return (kp,des)

class Painting:
  def __init__(self, name, room, descriptors):
    self.name = name
    self.room = room
    self.descriptors = descriptors

paintings = []
bf = cv.BFMatcher()

images = sorted(glob("/home/maartenvandenbrande/Documents/unif/computervisie/Database/*"))
#images = sorted(glob("/home/maartenvandenbrande/Documents/unif/computervisie/cutout_pictures/Zaal_D/*"))


def get_paintings():
	"""
	array = np.load("lepel.npy", allow_pickle=True)
	for element in array:
		p = Painting(element[0], element[1], element[2])
		paintings.append(p)
	"""
	index = 0
	for img in images:
		print("filling database: %0.2f percent" % (index/len(images)*100), end='\r')
		index += 1

		base=os.path.basename(img) 
		image_name = os.path.splitext(base)[0]
		image = cv.imread(img)

		kp1, descriptors = get_SIFTkeypoints(image)

		print(image_name)

		name = image_name
		room = "D"

		p = Painting(name, room, descriptors)
		paintings.append(p)
	print("filling database: 100 percent       ")
	print("done")

	for factor in range(50,100,1):
		total = 0
		dropped = 0
		index = 0
		for p1 in paintings:
			print("dropping descriptors: %0.2f percent" % (index/len(paintings)*100), end='\r')
			index += 1
			for p2 in paintings:
				if p1 != p2:
					matches = bf.knnMatch(p1.descriptors,p2.descriptors,k=2)

					for m,n in matches:
						total += 1
						if m.distance < factor/100*n.distance:
							np.delete(p1.descriptors, m.queryIdx)
							dropped += 1
		
		print("dropping descriptors: 100 percent       ")
		print("done")
		print("dropped: %s = %s" % (dropped, dropped/total*100))

		savearray = []
		for p in paintings:
			temp = [p.name, p.room, p.descriptors]
			savearray.append(np.array(temp))

		save = np.array(savearray)
		np.save("database_factor" + str(factor/100) + ".npy",save)
						
	"""
	print(paintings[0].descriptors.shape)
	for p in paintings:
		matches = bf.match(paintings[0].descriptors,p.descriptors)
		m = matches[1]
		print(matches[1])
		print("distance-- m: %s -- n: %s" % (m.distance, m.distance))
		print("imgIdx-- m: %s -- n: %s" % (m.imgIdx, m.imgIdx))
		print("queryIdx-- m: %s -- n: %s" % (m.queryIdx, m.queryIdx))
		print("trainIdx-- m: %s -- n: %s" % (m.trainIdx, m.trainIdx))
	"""


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

get_paintings()

cutouts = sorted(glob("/home/maartenvandenbrande/Documents/unif/computervisie/cutout_pictures/Zaal_D/*"))

for cutout in cutouts:
	img = cv.imread(cutout)

	t1 = time.time()
	pkeys, pdesc = get_SIFTkeypoints(img)

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
	print("1: zaal: %s, score: %s --- 2: zaal: %s, score: %s --- 3: zaal: %s, score: %s" % (list_name_totalscore[0][1], list_name_totalscore[0][2], list_name_totalscore[1][1], list_name_totalscore[1][2], list_name_totalscore[2][1], list_name_totalscore[2][2]))

	#cv.waitKey(0)