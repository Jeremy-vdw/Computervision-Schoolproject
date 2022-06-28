import cv2 as cv
from pymongo import MongoClient
import pickle
import os
from glob import glob
import feature_extraction as fe
import numpy as np

client = MongoClient('localhost',27017)
db = client["project_computervisie"]
paintings_db = db["paintings"]

"""Database import file

This file extract the painting features from the MONGO database and creates a Painting instance for each painting. 
Make sure that a mongodb is running on your local computer. Use get_paintings() to obtain all paintings.

"""

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

def __fill_paintings():
	print("Extract paintings from DB. May take a while.")
	for painting in paintings_db.find():
		name = painting["name"]
		room = painting["room"]
		histograms = pickle.loads(painting["histograms_rgb"])
		descriptors = pickle.loads(painting["descriptors"])
		p = Painting(name, room, histograms, descriptors)
		paintings.append(p)


def get_paintings():
	if(len(paintings) == 0):
		__fill_paintings()
	return paintings
