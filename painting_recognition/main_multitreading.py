import contour as c
import cv2
import threading
import time
import matching
import db_import as db
import HMM
import initialisation_room_probabilities as init_prob
import gui
import tkinter as tk
from tkinter import filedialog
import os
import sys

root = tk.Tk()
root.withdraw()

src = filedialog.askopenfilename(filetypes=[("video's", '*.mp4 *.avi'),('all files', '*')])

if (src == "") or (src == ()):
	exit()

#variabelen
#locatie video:
#src = "//home//maartenvandenbrande//Documents//unif//computervisie//video's//smartphone//MSK_05.mp4"
#src = "//home//maartenvandenbrande//Documents//unif//computervisie//video's//gopro_calibrated//MSK_14_undistored_with_M.avi"
#src = "/Users/jeremyvandewalle/telin.ugent.be/~dvhamme/computervisie_2021/videos/smartphone/MSK_03.mp4"


class contourThread (threading.Thread):
	def __init__(self, threadID, video_frame):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.video_frame = video_frame
		self.paintings = []
		self.found_painting_names = []
		self.deletable = False
	def run(self):
		time1 = time.time()
		#eruithalen van painting
		self.paintings = c.contour(self.video_frame)
		time2 = time.time()
		
		if len(self.paintings) == 0:
			self.deletable = True

		for painting in self.paintings:
			compare = matching.compare(painting)
			#compare = matching.compare_histograms(painting)
			
			if len(compare) != 0:
				#print("1:image: %s zaal: %s, score: %s --- 2:image: %s zaal: %s, score: %s --- 3:image: %s zaal: %s, score: %s" % (compare[0][0], compare[0][1], compare[0][2], compare[1][0], compare[1][1], compare[1][2], compare[2][0], compare[2][1], compare[2][2]))
				
				#cv2.imwrite("../Extractedpaintings/painting" + str(compare[0][0]) + "zaal" + str(compare[0][1]) + "kans" + str(compare[0][2]) + ".png", painting)
				
				if 0.45*compare[0][2] > compare[1][2]:
					self.found_painting_names.append((compare[0][0], compare[0][1]))
					print("1:image: %s zaal: %s, verschill in score: %0.5f, tijd: 1) %0.2f -- 2) %0.2f " % (compare[0][0], compare[0][1], compare[1][2]/compare[0][2], (time2-time1), (time.time()-time2)))
				else:
					cv2.imwrite("../Extractedpaintings/painting" + str(compare[0][0]) + "zaal" + str(compare[0][1]) + "kans" + str(compare[0][2]) + ".png", painting)
					print("thrown -- 1:image: %s zaal: %s, verschill in score: %0.5f, tijd: 1) %0.2f -- 2) %0.2f " % (compare[0][0], compare[0][1], compare[1][2]/compare[0][2], (time2-time1), (time.time()-time2)))


class ThreadedCamera(object):
	def __init__(self, src=0):
		self.capture = cv2.VideoCapture(src)
		self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

		self.pause = False

		#resize
		scale_percent = 10 # percent of original size
		self.width = self.capture.get(3) #get width
		self.height = self.capture.get(4) #get height
		self.resize_dim = (int(self.width * scale_percent / 100), int(self.height * scale_percent / 100))

		#laplacian
		self.bestLaplacian = 0
		self.bestImage = []

		#dubble check
		self.recent_images = [None] * 30
		self.recent_images_index = 0

		# FPS = 1/X
		# X = desired FPS
		self.FPS = 1/self.capture.get(cv2.CAP_PROP_FPS)
		self.FPS_MS = int(self.FPS * 1000)
		self.newframe = False

		#frame tracker
		self.frame_limit_number = int(2*self.capture.get(cv2.CAP_PROP_FPS)/30)
		self.frame_limit = 1 # over hoeveel frames gekeken wordt voor matching
		self.frame_number = 0

		#contourthreads
		self.contourThreads = []
		self.maxContourThreads = 6
		self.contourThreadIdCounter = 0
		self.paintingindex = 0

		#frames missed check
		self.total_frames = 0
		self.frames_checked = 0

		#HMM
		self.room_probs = init_prob.room_probs
		self.highestscore_rooms = [""]

		# Start frame retrieval thread
		self.thread = threading.Thread(target=self.update, args=())
		self.thread.daemon = True
		self.thread.start()


	def update(self):
		#haalt elke 1/30 seconden (=33ms) de volgende frame en doet dit op een aparte thread zodat er meer tijd is voor het verwerken van de frames op de main thread
		while True:
			if self.capture.isOpened() and (self.pause == False):
				(self.status, self.frame) = self.capture.read()
				self.newframe = True
				self.total_frames += 1

			if self.status == False:
				os.execl(sys.executable, sys.executable, *sys.argv)

			time.sleep(self.FPS)

	def show_frame(self):
		key = cv2.waitKey(1)

		if key == 112: #p
			self.pause = not self.pause

		if key == 113: #q
			exit()

		if key == 114: #r
			os.execl(sys.executable, sys.executable, *sys.argv)
		
		if self.newframe:
			self.newframe = False

			gui.show_room(self.highestscore_rooms, self.frame)

			#resizen van de frame om de laplacian sneller te laten gaan en vervolgens de varience van de laplace te berekenen en te bekijken of deze het hoogste is van de afgelopen frames.
			#kort gezegd dit zoekt naar de minst blurrige foto van de afgelopen frames
			resized = cv2.resize(self.frame, self.resize_dim, interpolation = cv2.INTER_AREA)
			laplacian = cv2.Laplacian(resized, cv2.CV_32F).var()
			if laplacian > self.bestLaplacian:
				self.bestLaplacian = laplacian
				self.bestImage = self.frame

			#als de frame_limit is berijkt wordt er gekeken of er plaats is in de thread array indien dit het geval is wordt er een nieuwe thread aangemaakt die de beste frame verwerkt
			if self.frame_number >= self.frame_limit:
				if len(self.contourThreads) < self.maxContourThreads + 1:
					thread = contourThread(self.contourThreadIdCounter, self.bestImage)
					thread.setDaemon(True)
					thread.start()
					self.contourThreads.append(thread)
					self.contourThreadIdCounter += 1
					self.bestLaplacian = 0
					self.bestImage = []
					self.frame_number = 0

			self.frame_number = self.frame_number + 1
			self.frames_checked += 1
		

		#bekijkt of er in de array van threads dat de contouren zoekt een klaar is en delete deze dan
		for thread in self.contourThreads:
			#checken of er een nieuwe image is en indien het geval opslaan self.recent_images houd de laatste 30 paintings bij
			if not thread.is_alive():
				#for painting in thread.paintings:
					#cv2.imshow("Extracted ", painting)
					#cv2.imshow("Extracted " + str(self.paintingindex), painting) #uncomment dit als je alle gevonden schilderijen wilt zien (warning maakt heel veel kaders)
					#self.paintingindex += 1
					
				for (name, room) in thread.found_painting_names:
					self.room_probs = HMM.room_given_painting_probs(self.room_probs, name, room)
					self.highestscore_rooms = sorted(self.room_probs, key=self.room_probs.get, reverse=True)[:5]
					print("1: %s : %0.3f --- 2: %s : %0.3f --- 3: %s : %0.3f --- 4: %s : %0.3f --- 5: %s : %0.3f" % (self.highestscore_rooms[0],self.room_probs[self.highestscore_rooms[0]],self.highestscore_rooms[1],self.room_probs[self.highestscore_rooms[1]],self.highestscore_rooms[2],self.room_probs[self.highestscore_rooms[2]],self.highestscore_rooms[3],self.room_probs[self.highestscore_rooms[3]],self.highestscore_rooms[4],self.room_probs[self.highestscore_rooms[4]]))# , end='\r')

				thread.deletable = True
		self.contourThreads = [t for t in self.contourThreads if not t.deletable]
		
		self.frame_limit = (len(self.contourThreads) + 1) * self.frame_limit_number

		print("number of threads: %s -- frames missed: %s" % (len(self.contourThreads), self.total_frames - self.frames_checked), end='\r')

		

# get db in mem => als True megegeven wordt vult hij memory met berekende waarden indien niets of false wordt meegegeven gebruikt hij db
db.get_paintings()

probs = []

threaded_camera = ThreadedCamera(src)
while True:
	threaded_camera.show_frame()