import groundthruth_read as gt 
import os
import cv2 as cv
import numpy as np 
from glob import glob
import contour
import time
import feature_extraction as fe
import db_import as db
import matching
import csv

groundtruth_paintings = gt.get_groundtruth_paintings()

def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)

zalen = sorted(list(set(b.room for b in groundtruth_paintings)))

paintings_db = db.get_paintings()


with open('matching_compare.csv', 'w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(["zaal", "paintings in room", "paintings correct RGB", "paintings correct SIFT", "average time RGB", "average time SIFT"])
    for zaal in zalen:

        imagepath_folder = sorted(glob("/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Computervisie 2020 Project Database/dataset_pictures_msk/"+ zaal + "/*.jpg"))
        print(zaal)

        paintings_room = 0
        correct_room_hist = 0
        correct_room_sift = 0
        avg_time_hist = 0
        avg_time_sift = 0

        for imagepath in imagepath_folder:

            image = cv.imread(imagepath)
            image = resize_image_to_width(image, int(image.shape[1]/4))
            paintings = contour.contour(image)

            results = []
            
            for painting in paintings:

                paintings_room += 1

                histograms = fe.get_part_histograms(painting)

                results = []        

                t1 = time.time()
                for painting2 in paintings_db:

                    score = painting2.compare_with_histograms(histograms)
                    results.append((painting2.room, score, painting2.name))

                t2 = time.time()
                avg_time_hist += (t2 - t1)

                results = sorted(results, key = lambda x: x[1], reverse = False)

                if ("Zaal_" + results[0][0]).lower() == (zaal).lower():
                    correct_room_hist += 1
                    
                ##sift    

                t1 = time.time()

                results = matching.compare(painting)

                t2 = time.time()
                avg_time_sift += (t2 - t1)

                if len(results) > 0:
                    if ("Zaal_" + results[0][1]).lower() == (zaal).lower():
                        correct_room_sift += 1

                '''
                cv.imshow("image", painting)d
                cv.waitKey(0)
                cv.destroyAllWindows()
                '''

        if paintings_room > 0:
            avg_time_hist = avg_time_hist / paintings_room
            avg_time_sift = avg_time_sift / paintings_room

        print("Paintings in room: " + str(paintings_room))
        print("Correct detections in room with RGB: " + str(correct_room_hist))
        print("Correct detections in room with SIFT: " + str(correct_room_sift))
        print("Average time /painting with RGB: " + str(avg_time_hist))
        print("Average time /painting with SIFT: " + str(avg_time_sift))

        writer.writerow([zaal, paintings_room, correct_room_hist, correct_room_sift, avg_time_hist, avg_time_sift])


