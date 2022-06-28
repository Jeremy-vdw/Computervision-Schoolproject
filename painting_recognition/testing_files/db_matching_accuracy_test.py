import time
from glob import glob
import numpy as np
import os
import contour
import feature_extraction as fe
import feature_compare as fc
import matching
import cv2 as cv
import db_import

class Painting:
	def __init__(self, name, room, histograms, descriptors):
		self.name = name
		self.room = room
		self.histograms = histograms
		self.descriptors = descriptors

	def compare_with_histograms(self, histograms):
		histogram_part_score = 0
		for y in range(0, len(self.histograms)):
			for x in range(0, len(self.histograms[y])):
				histogram_part_score += cv.compareHist(self.histograms[y][x], histograms[y][x], cv.HISTCMP_BHATTACHARYYA)
		histogram_part_score /= len(self.histograms)*len(self.histograms)

		return histogram_part_score


paintings = []

url = sorted(glob("/home/maartenvandenbrande/Documents/unif/computervisie/dataset_pictures_msk/*"))
urldescriptors = "//home//maartenvandenbrande//Documents//unif//computervisie//tests//lessdescriptors300//"

databaseurl = "database.npy"


database_nummer = 20
avrg_matching_score = 0
avrg_normalized_matching_score = 0
totaal = 0
database_contours_found = 0
juiste_zaal = 0

database_start_time = time.time()

paintings_hist = db_import.get_paintings(False)

array = np.load(databaseurl, allow_pickle=True)
print("database nummer: %s" % database_nummer)
database_avg_match_time = (0,0) #(number of maches, total match time)

for element in paintings_hist:
	try:
		temp = array[np.where(array[:,0]==element.name),4][0][0][0][0]
		p = Painting(element.name, element.room, element.histograms, temp)
		paintings.append(p)
	except:
		print("fail")

progress = 0
for zaal in url:
	avrg_matching_score_zaal = 0
	avrg_normalized_matching_score_zaal = 0
	totaal_zaal = 0
	database_contours_found_zaal = 0
	juiste_zaal_zaal = 0
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
		image = fe.resize_image_to_width(image, 300)
		#cv.imshow("test", image)
		contours = contour.contour(image)
		#print("number: %s" % (len(contours)))
		#cv.waitKey(0)
		#exit()

		zaal_juist = False

		#t2 = time.time()
		found_image_score = (0, 0) #(normalized, score)
		for contour_image in contours:
			start_time_contour = time.time()

			#histogram_rgb = fe.get_rgbhistograms(contour_image)
			#histogram_lbp = fe.get_LBPhistogram(contour_image)
			pkeys, pdesc = fe.get_SIFTkeypoints(contour_image)
			histograms = fe.get_part_histograms(contour_image)

			results = []
			max_keypoints = 0

			for image in paintings:
				#rgb_histogram_score = fc.compare_rgbhistograms(image.histogram_rgb, histogram_rgb)
				#lbp_histogram_score = fc.compare_lbphistogram(image.histogram_lbp, histogram_lbp)
				keypoints = fc.compare_descriptors(image.descriptors, pdesc)
				histograms_score = 1-image.compare_with_histograms(histograms)


				#list_name_feature_scores.append((image.name, image.room, rgb_histogram_score, lbp_histogram_score, keypoints))
				results.append((image.name, image.room, histograms_score, keypoints))
				
				if keypoints > max_keypoints:
					max_keypoints = keypoints
				
			
			if max_keypoints == 0:
				continue
			
			
			#list with total scores of each image 
			score_painting = 0
			total_score = 0
			
			for name_score in results:
				#calculated_score = matching.calculate_score(name_score[2], name_score[3], name_score[4], max_keypoints)
				#calculated_score = name_score[2] / max_keypoints
				calculated_score = matching.calculate_score_keypoints_histograms(name_score[2], name_score[3], max_keypoints)

				name_score = (name_score[0], name_score[1], calculated_score)

				total_score += calculated_score
				if name_score[0].split("-")[0] == name:
					score_painting = calculated_score

			best = sorted(results, key = lambda x: x[2], reverse = True)[0]
				
			if best[1] == zaal.split("_")[-1]:
				zaal_juist = True

			normalized_score = score_painting/total_score

			if normalized_score > found_image_score[0]:
				found_image_score = (normalized_score, score_painting)

			database_avg_match_time = (database_avg_match_time[0]+1, database_avg_match_time[1] + (time.time()-start_time_contour))

		if zaal_juist:
			juiste_zaal_zaal += 1
			juiste_zaal += 1

		#t3 = time.time()
		if(len(contours)>0):
			database_contours_found += 1
			database_contours_found_zaal += 1
			avrg_normalized_matching_score += found_image_score[0]
			avrg_normalized_matching_score_zaal += found_image_score[0]
			avrg_matching_score += found_image_score[1]
			avrg_matching_score_zaal += found_image_score[1]
			
		
		#print("time --- init: %0.5f --- contour: %0.5f --- match: %0.5f --- end: %0.5f" % (t1-t0,t2-t1,t3-t2,time.time()-t3))

		totaal += 1
		totaal_zaal += 1

	print("zaal: " + zaal.split("/")[-1] + "                 ")
	if database_contours_found_zaal != 0:
		print("painting found: %s or %0.3f percent -- avg match score: %0.5f percent -- avg normalized match score: %0.5f percent --- juiste zaal: %0.5f percent" % (database_contours_found_zaal, database_contours_found_zaal/totaal_zaal*100, avrg_matching_score_zaal/(database_contours_found_zaal)*100, avrg_normalized_matching_score_zaal/(database_contours_found_zaal)*100, juiste_zaal_zaal/database_contours_found_zaal *100))
	else:
		print("no paintings found")

	progress += 1
#print(database_avg_match_time[1]/database_avg_match_time[0])
print("painting found: %s or %0.3f percent -- avg match score: %0.10f percent -- avg normalized match score: %0.10f percent --- juiste zaal: %0.5f percent" % (database_contours_found, database_contours_found/totaal*100, avrg_matching_score/(database_contours_found)*100, avrg_normalized_matching_score/(database_contours_found)*100, juiste_zaal/database_contours_found *100))
print("in %0.2f seconden with avg match time of: %0.5f" % (time.time()-database_start_time, database_avg_match_time[1]/database_avg_match_time[0]))

