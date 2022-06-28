from glob import glob
import cv2
import os
import numpy as np
import contour as c

#url = "//home//maartenvandenbrande//Documents//unif//computervisie//dataset_pictures_msk//zaal_16//"
url = "//home//maartenvandenbrande//Documents//unif//computervisie//test_pictures_msk//"
url2 = "//home//maartenvandenbrande//Documents//unif//computervisie//cutout_pictures//"

def on_trackbar(var):
	kernel = cv2.getTrackbarPos("kernel","original")*2+1
	sigma = cv2.getTrackbarPos("sigma","original")/10
	sigma2 = cv2.getTrackbarPos("sigma2","original")/10
	thresh_1 = cv2.getTrackbarPos("thresh_1","original")
	thresh_2 = cv2.getTrackbarPos("thresh_2","original")
	verschiltresh = cv2.getTrackbarPos("verschiltresh","original")
	dilatekernel = cv2.getTrackbarPos("dilatekernel","original")
	sobbelkernel = cv2.getTrackbarPos("sobbelkernel","original")*2+1

	retreaved = c.contour(img, (kernel, kernel), sigma, thresh_1, thresh_2, (dilatekernel, dilatekernel), verschiltresh, sobbelkernel)
	index = 0
	for retreave in retreaved:
		cv2.imshow("lepel " + str(index), retreave)
		index += 1

cv2.namedWindow("original")
cv2.createTrackbar("kernel", "original" , 2, 10, on_trackbar)
cv2.createTrackbar("sigma", "original", 12, 100, on_trackbar)
cv2.createTrackbar("sigma2", "original", 12, 100, on_trackbar)
cv2.createTrackbar("thresh_1", "original", 42, 255, on_trackbar)
cv2.createTrackbar("thresh_2", "original", 170, 255, on_trackbar)
cv2.createTrackbar("verschiltresh", "original", 80, 255, on_trackbar)
cv2.createTrackbar("dilatekernel", "original", 3, 10, on_trackbar)
cv2.createTrackbar("sobbelkernel", "original", 3, 10, on_trackbar)

for name in os.listdir(url):
	img = cv2.imread(url + name)
	img = cv2.resize(img, dsize=(int(img.shape[1]/5), int(img.shape[0]/5)))
	cv2.imshow("original"+name, img)
	
	on_trackbar(0)
	key = cv2.waitKey(0)
	if(key == 113):
		exit(0)


cv2.destroyAllWindows()

"""
for zaalname in os.listdir(url):
	#cv2.imshow("original", img)
	originaldir = url + zaalname + "//"
	newdir = url2 + zaalname + "//"
	os.mkdir(url2 + zaalname)
	index = 0
	for imgname in os.listdir(originaldir):
		img = cv2.imread(originaldir + imgname)
		img = cv2.resize(img, dsize=(int(img.shape[1]/5), int(img.shape[0]/5)))

		retreaved = c.contour(img)
		for retreave in retreaved:
			cv2.imwrite(newdir + "image_" + str(index) + ".png", retreave)
			index += 1
"""