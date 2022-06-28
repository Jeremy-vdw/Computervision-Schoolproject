import groundthruth_read as gt 
import os
import cv2 as cv
import numpy as np 
from glob import glob
import contour
import time
import feature_extraction as fe
import db_import as db
import matplotlib.pyplot as plt

groundtruth_paintings = gt.get_groundtruth_paintings()

def resize_image_to_width(image, width):
	height = int(width / image.shape[1] * image.shape[0])
	dimensions = (width, height)
	return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

'''
def get_paintings():
	db = np.load('db_test.npy', allow_pickle=True)
	return db
'''

zalen = sorted(list(set(b.room for b in groundtruth_paintings)))

paintings_db = db.get_paintings()

for zaal in zalen:

	imagepath_folder = sorted(glob("/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Computervisie 2020 Project Database/dataset_pictures_msk/"+ zaal + "/*.jpg"))
	print(zaal)

	paintings_room = 0
	correct_room = 0
	avg_time = 0

	for imagepath in imagepath_folder:

		image = cv.imread(imagepath)
		image = resize_image_to_width(image, int(image.shape[1]/4))
		paintings = contour.contour(image)

		results = []
		
		for painting in paintings:

			paintings_room += 1

			histograms = fe.get_part_histograms(painting)

			results = []

			'''
			cv.imshow("image", painting)
			cv.waitKey(0)
			cv.destroyAllWindows()
			'''

			t1 = time.time()
			for painting2 in paintings_db:

				score = painting2.compare_with_histograms(histograms)
				results.append((painting2.room, score))

			t2 = time.time()
			avg_time += (t2 - t1)

			results = sorted(results, key = lambda x: x[1], reverse = False)
			if ("Zaal_" + results[0][0]).lower() == (zaal).lower():
				correct_room += 1
	
	if paintings_room > 0:
		avg_time = avg_time / paintings_room

	print("Paintings in room: " + str(paintings_room))
	print("Correct detections in room: " + str(correct_room))
	print("Average time /painting: " + str(avg_time))

