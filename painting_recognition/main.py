import contour as c
import cv2
import numpy as np
import gui
import screeninfo
import os
from glob import glob
import matching
import db_import as db

# img = cv2.imread("painting_recognition\\test_images\\img2.png")
# resized = gui.resizeImage(img, dimension=(960, 540))
# if(img is not None):
#     cv2.imshow("test", resized)
# else:
#     print("Geen img gevonden (check working directory)")
# cv2.waitKey(0)
# img_cutout = c.contour(img)
# cv2.imshow("contour", img_cutout)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# video
video_path = "//home//maartenvandenbrande//Documents//unif//computervisie//video's//smartphone//MSK_03.mp4"
#video_path = "/Users/jeremyvandewalle/Documents/UGent/telin.ugent.be/~dvhamme/computervisie_2021/videos/smartphone/MSK_01.mp4"

#videos = sorted(glob("test_video/MSK*"))
#video_path = videos[0]

base=os.path.basename(video_path) # remove pathname
naam = os.path.splitext(base)[0] # get filename without extension

video_capture = cv2.VideoCapture(video_path)
frame_limit = 20 # over hoeveel frames gekeken wordt voor matching
(frame_number, frame_total) = (0, 0)
counter = 0

# get db in mem => als True megegeven wordt vult hij memory met berekende waarden indien niets of false wordt meegegeven gebruikt hij db
db.get_paintings()

# initialize windows 
monitor = screeninfo.get_monitors()[0]
width, height = (monitor.width, monitor.height)
gui.createWindowAtCoordinates("Video", int(width/2), 0)
matchWindowFlag = 0  # is used to determine which match window is used.

#laplaciaan of beste frame te kiezen
bestLaplacian = 0
bestImage = []
scale_percent = 20 # percent of original size
resize_dim = (int(video_capture.get(3) * scale_percent / 100), int(video_capture.get(4) * scale_percent / 100))

#recent images
recent_images = [None] * 30
recent_images_index = 0

while(video_capture.isOpened() and not cv2.waitKey(33)==ord('q')):
	video_frame = video_capture.read()[1] # sla de eerste over
	#video_frame = gui.resizeImage(video_frame, dimension = (int(width/2), height))

	gui.showImage("Video", video_frame, 1)
	
	resized = cv2.resize(video_frame, resize_dim, interpolation = cv2.INTER_AREA)
	laplacian = cv2.Laplacian(resized, cv2.CV_32F).var()
	if laplacian > bestLaplacian:
		bestLaplacian = laplacian
		bestImage = video_frame


	if cv2.Laplacian(video_frame, cv2.CV_32F).var() > bestLaplacian:
		bestImage = video_frame

	# pick out frames
	if frame_number == frame_limit:
		# vind schilderij met contour.py in huidige frame
		paintings = c.contour(bestImage)

		# schrijf weg als afbeelding
		"""
		if not os.path.exists("Extracted images/" + naam):
		    os.makedirs("Extracted images/" + naam)
		filename = f"video {naam} frame {counter}"
		cv2.imwrite("Extracted images/" + naam + "/" + filename + ".png", painting)
		"""

		paintingindex = 0
		for painting in paintings:
			"""
			gui.showImage("Extracted " + str(paintingindex), painting, 1)
			paintingindex += 1
			filename = f"video {naam} frame {counter}"
			cv2.imwrite("Extracted images/" + naam + "/" + filename + ".png", painting)
			"""
			#check of painting al is geweest indien niet => uitvoeren van keypoint compare en opslaan in recent images deze houdt de laatste 30 images bij (de histogrammen)
			recent_image = matching.compare_recent(painting, recent_images)
			if not recent_image == None:
				recent_images[recent_images_index] = recent_image
				if recent_images_index == 29:
					recent_images_index = 0
				else:
					recent_images_index += 1
				temp = matching.compare(painting)
				print("1: zaal: %s, score: %s --- 2: zaal: %s, score: %s --- 3: zaal: %s, score: %s" % (temp[0][1], temp[0][2], temp[1][1], temp[1][2], temp[2][1], temp[2][2]))
		

		frame_number = 0
		bestLaplacian = 0
		bestImage = []
	
	counter += 1
	frame_number = frame_number + 1
	frame_index = frame_total + 1

video_capture.release()

cv2.destroyAllWindows()