import cv2 as cv
import numpy as np
import os
import gui
from glob import glob
from matplotlib import pyplot as plt
import feature_extraction as fe
import pymongo
import pickle

#No need to run this file as the docker is filled with the JSON after running docker-compose 

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["project_computervisie"]
paintings = db["paintings"]
#paintings_descriptors = db["paintings_descriptors"] #not yet 

#clear collections
paintings.delete_many({})
print(paintings.count())

images = sorted(glob("/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Database/*"))

c = 0
for img in images:

    base=os.path.basename(img) 
    image_name = os.path.splitext(base)[0]
    image_name = image_name.replace("IMG_", "")
    image_splitted = image_name.split('_')
    image_gray = cv.imread(img, 0)
    image = cv.imread(img)

    name = image_splitted[4] + '-' + image_splitted[6]
    room = image_splitted[1]

    histogram_rgb = fe.get_part_histograms(image)

    descriptors = None

    temp = np.load("database.npy", allow_pickle=True)
    for element in temp:
        if(name == element[0] and room == element[1]):
	        descriptors = element[4][0][0]



    image_db = {
        "name" : name,
        "room": room,
        "histograms_rgb": pickle.dumps(histogram_rgb),
        "descriptors": pickle.dumps(descriptors)
    }

    dump = paintings.insert_one(image_db)

    print(c)
    c += 1

print(paintings.count())

