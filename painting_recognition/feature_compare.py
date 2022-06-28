import cv2 as cv
import numpy as np
import time

bf = cv.BFMatcher()

def matches_ratiotest(matches):
	# Apply ratio test
	if(len(matches) <= 1) or (len(matches[0]) != 2):
		return 0
	good = []
	for m,n in matches:
		if m.distance < 0.60*n.distance:
			good.append([m])
	return len(good)

def compare_rgbhistograms(histogram_base, histogram_compare):
	return cv.compareHist(np.array(histogram_base), np.array(histogram_compare), 0)

def compare_lbphistogram(histogram_base, histogram_compare):
	return cv.compareHist(np.array(histogram_base, dtype=np.float32), np.array(histogram_compare, dtype=np.float32), 0)

def compare_descriptors(descriptors_base, descriptors_compare):
	# BFMatcher with default params
	if descriptors_base is None or descriptors_compare is None or descriptors_base.size == 0 or descriptors_compare.size == 0:
		return 0
	
	matches = bf.knnMatch(descriptors_base,descriptors_compare,k=2)
	
	keypoints = matches_ratiotest(matches)
	return keypoints
	
def comparison_histograms(histograms_base, histograms_compare):
	
	histogram_part_score = 0
	for y in range(0, len(histograms_base)):
		for x in range(0, len(histograms_base[y])):
			histogram_part_score += cv.compareHist(histograms_base[y][x], histograms_compare[y][x], cv.HISTCMP_BHATTACHARYYA)
	histogram_part_score /= len(histograms_base)*len(histograms_base)

	return histogram_part_score
