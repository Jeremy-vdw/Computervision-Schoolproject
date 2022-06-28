import groundthruth_read as gt 
import os
import cv2 as cv
import numpy as np 
from glob import glob
from shapely.geometry import Polygon
import recognize_other_approach as recognize
import contour

### Testing groundthruth average recognition 

groundtruth_paintings = gt.get_groundtruth_paintings()

def calculate_accuracy(pts1, pts2):
    tp = Polygon(pts1)
    pp = Polygon(pts2)
    if pp.is_valid :
        intersection = tp.intersection(pp)
        if(intersection == 0):
            return 0
        union = tp.area + pp.area - intersection.area
        return intersection.area / union
    return 0


def resize_image_to_width(image, width):
    height = int(width / image.shape[1] * image.shape[0])
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation=cv.INTER_AREA)


def draw_polylines(image, pts, color):
    pts = pts.reshape((-1,1,2))
    cv.polylines(image,[pts],True,color, 5)

zalen = sorted(list(set(b.room for b in groundtruth_paintings)))

global_accuracy = 0

'''
for zaal in zalen:

    gtp_zaal = list(filter(lambda x: x.room == zaal, groundtruth_paintings))
    imagepath_folder = sorted(glob("/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/Computervisie 2020 Project Database/dataset_pictures_msk/"+ zaal + "/*.jpg"))


    true_positive = 0
    false_negatives = 0
    total_accuracy = 0

    for imagepath in imagepath_folder:
        base=os.path.basename(imagepath) 
        image_name = os.path.splitext(base)[0]
        image = cv.imread(imagepath)
        image = resize_image_to_width(image, int(image.shape[1]/4))
        #approx = recognize.detect_contours(image)
        approx = contour.contour_pointers(image)

        gtp_name = list(filter(lambda x: x.imagename.strip() == image_name.strip(), gtp_zaal))

        for gt in gtp_name:
            pts = np.array([gt.tl/4, gt.tr/4, gt.br/4, gt.bl/4], np.int32)
            draw_polylines(image, pts, (255, 0, 0))
            #calculate TP & FN
            for app in approx:
                accuracy = calculate_accuracy(pts, app.reshape(4,2))
                if(accuracy > 0.5): #detected
                    true_positive += 1
                    total_accuracy += accuracy

        for app in approx:
            draw_polylines(image, app, (0, 0, 255))

        not_detected = len(gtp_name) - len(approx)
        #print("Not detected : " + str(not_detected))
        false_negatives += not_detected

        cv.imshow("Image " + image_name, image)
        cv.waitKey(1)

    cv.waitKey(0)
    cv.destroyAllWindows()

    average_accuracy = total_accuracy / true_positive

    print("Average accuracy of " + zaal + " : " + str(average_accuracy))
    print("Amount of paintins not detected in " + zaal + " : " + str(false_negatives))

    global_accuracy += average_accuracy


global_average = global_accuracy / len(zalen)
print(global_average)
'''
    
