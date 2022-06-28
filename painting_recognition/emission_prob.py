#import groundthruth_read as gt 
import os
import cv2 as cv
import numpy as np 
from glob import glob
#from shapely.geometry import Polygon
import contour
import matching as m
import db_import as db
import time
import csv

db.get_paintings(True)

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

url_db = sorted(glob('/home/maartenvandenbrande/Documents/unif/computervisie/Database/*'))
url_pictures = sorted(glob('/home/maartenvandenbrande/Documents/unif/computervisie/dataset_pictures_msk/*'))
#array = np.load("emission_prob.npy", allow_pickle=True)
#print(array)

dict_zalen = {}

totaal = 0

for imagepath in url_db:
	image = cv.imread(imagepath)
	splited = os.path.basename(imagepath).split("_")
	zaal = "Zaal_" + splited[1]
	name = splited[-3] + "-" + splited[-1].split(".")[0]
	dict_zalen[zaal] = {}

for imagepath in url_db:
	image = cv.imread(imagepath)
	splited = os.path.basename(imagepath).split("_")
	zaal = "Zaal_" + splited[1]
	name = splited[-3] + "-" + splited[-1].split(".")[0]

	som = 0
	compare = m.compare(image)
	if (len(compare) != 0):
		for element in compare:
			som += element[2]**2
		for element in compare:
			if zaal + "_" + name in dict_zalen["Zaal_" + element[1]]:
				dict_zalen["Zaal_" + element[1]][zaal + "_" + name] += (element[2])**(2)/som
			else:
				dict_zalen["Zaal_" + element[1]][zaal + "_" + name] = (element[2])**(2)/som
	else:
		print("no compare found- name: " + name)

	#print(dict_zalen)

	totaal += 1
	print("%0.3f percent" % (totaal/800*100), end="\r")


"""
aantal = 0
totaal = 0

for zaal_path in url_pictures:
	for imagepath in sorted(glob(zaal_path + "/*")):
		image = cv.imread(imagepath)
		zaal = os.path.basename(zaal_path)
		splited = os.path.basename(imagepath).split("_")
		name = splited[-1].split(".")[0]
		image = resize_image_to_width(image, 300)
		paintings = contour.contour(image)
		juist = False
		for painting in paintings:
			temp = m.compare(painting)
			if (len(temp) != 0):
				if(name == temp[0][0].split("-")[0]):
					juist = True
					db_naam = temp[0][0]
		if juist:
			dict_zalen[zaal][db_naam] += 1
		totaal += 1
		print("%0.3f percent" % (totaal/533*100), end="\r")
"""
np.save('emission_prob_2.npy', dict_zalen, allow_pickle=True)
exit()

groundtruth_paintings = gt.get_groundtruth_paintings()

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

zalen = sorted(list(set(b.room for b in groundtruth_paintings)))

dict_zalen = {}

for zaal in zalen:

        imagepath_folder = sorted(glob("/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Computervisie 2020 Project Database/dataset_pictures_msk/"+ zaal + "/*.jpg"))
        print(zaal)

        dict_zaal = {}

        for imagepath in imagepath_folder:

            image = cv.imread(imagepath)
            image = resize_image_to_width(image, int(image.shape[1]/4))
            paintings = contour.contour(image)
           
            for painting in paintings:

                list_scores = m.compare(painting)
                if(len(list_scores) > 0):
                    if list_scores[0][0] in dict_zaal:
                        dict_zaal[list_scores[0][0]] += 1
                    else:  
                        dict_zaal[list_scores[0][0]] = 1

        dict_zalen[zaal] = dict_zaal
        print("Zaal " + str(zaal) +  ": " + str(dict_zaal))

print(dict_zalen)
np.save('emission_prob.npy', dict_zalen, allow_pickle=True)

                



                
            





