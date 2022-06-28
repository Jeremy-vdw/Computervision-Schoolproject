import cv2 as cv
import feature_extraction as fe
import feature_compare as fc
from glob import glob
import numpy as np
import gui
import db_import as db
import time
images_to_compare = sorted(glob("/Users/jeremyvandewalle/Github/project_computervisie/Extractedpainting/*.png"))

images_db_path = sorted(glob("/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Database/*"))

images_db = db.get_paintings()




max_keypoints = 0
img_score = []


def calc_score(rgb_score, lbp_score, keypoints, max_keypoints):
    keypoints = keypoints / max_keypoints
    rgb_weight = 6
    lbp_weight = 5
    keypoints_weight = 5
    return (rgb_score * rgb_weight + lbp_score * lbp_weight + keypoints * keypoints_weight) / (rgb_weight + lbp_weight + keypoints_weight)

for image in images_to_compare:

    image = cv.imread(image)

    #calculate hists
    histograms = fe.get_rgbhistograms(image)
    lbp = fe.get_LBPhistogram(image)
    kp2, des2 = fe.get_SIFTkeypoints(image)

    #parameters
    max_keypoints = 0
    img_score = []

    #compare to db
    for image_db in images_db:


        rgb_histogram_score = fc.compare_rgbhistograms(image_db.histogram_rgb, histograms)
        lbp_histogram_score = fc.compare_lbphistogram(image_db.histogram_lbp, lbp)
        keypoints = fc.compare_descriptors(image_db.descriptors, des2)
        img_score.append((image_db.name, rgb_histogram_score, lbp_histogram_score, keypoints))

        if(keypoints > max_keypoints): 
            max_keypoints = keypoints

    gui.showImage("scanned image ", image)

    img_calcscore = []
    print("max keypoints:" + str(max_keypoints))

    for score in img_score:
        calculated_score = calc_score(score[1], score[2], score[3], max_keypoints)
        img_calcscore.append((score[0], calculated_score, score[1], score[2], score[3]))

    img_calcscore = sorted(img_calcscore, key = lambda x: x[1], reverse = True)[:10]

    for scores in img_calcscore:
        print("total score: " + str(scores[1]) + " --> RGB score : " + str(scores[2]) + ", LBP score : " + str(scores[3]) + ", Keypoints: " + str(scores[4]))



