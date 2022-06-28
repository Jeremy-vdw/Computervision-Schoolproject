# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:14:18 2021

@author: Steven
"""
import numpy as np
from glob import glob
import cv2
import os

calibration_files = sorted(glob("*.npz"))
images = sorted(glob("calibration_images_calibration_M/*.png"))
FPS = 5
crop = 0.5

frame = cv2.imread(images[0])
size = (frame.shape[1], frame.shape[0])

distCoeff = []
intrinsic_matrix = []

for calibration_file in calibration_files:
    
    print ('Loading calibration file ' + calibration_file)
        
    npz_calib_file = np.load(calibration_file)
    
    distCoeff.append(npz_calib_file['distCoeff'])
    intrinsic_matrix.append(npz_calib_file['intrinsic_matrix'])
    
    print(npz_calib_file['distCoeff'])
    print(npz_calib_file['intrinsic_matrix'])
    
    
    npz_calib_file.close()
    
    print('Finished loading file')
    print(' ')

print("calibration M")
print(distCoeff[0])
print(intrinsic_matrix[0])

print("calibration W")
print(distCoeff[1])
print(intrinsic_matrix[1])

codec = cv2.VideoWriter_fourcc(*'XVID')
video_out_M = cv2.VideoWriter('test_video_calibration_M.avi', codec, FPS, size, 1) # 1 signifies isColor == True
video_out_W = cv2.VideoWriter('test_video_calibration_W.avi', codec, FPS, size, 1)

newMat_M, ROI_M = cv2.getOptimalNewCameraMatrix(intrinsic_matrix[0], distCoeff[0], size, alpha = crop, centerPrincipalPoint = 1)
mapx_M, mapy_M = cv2.initUndistortRectifyMap(intrinsic_matrix[0], distCoeff[0], None, newMat_M, size, m1type = cv2.CV_32FC1)

newMat_W, ROI_W = cv2.getOptimalNewCameraMatrix(intrinsic_matrix[1], distCoeff[1], size, alpha = crop, centerPrincipalPoint = 1)
mapx_W, mapy_W = cv2.initUndistortRectifyMap(intrinsic_matrix[1], distCoeff[1], None, newMat_W, size, m1type = cv2.CV_32FC1)

print("Undistorting...")

x = 0.1
for i in range(int(1/x)):
    print("-", end = "")

print("")

counter = 0
total = len(images)
for img in images:
    counter += 1
    if (counter >= x*total):
        print('\u2588', end="")     
        x += 0.1
            
    image = cv2.imread(img)
    # video_out_M.write(image)
    # video_out_W.write(image)

    dst_M = cv2.remap(image, mapx_M, mapy_M, cv2.INTER_LINEAR)
    dst_W = cv2.remap(image, mapx_W, mapy_W, cv2.INTER_LINEAR)
    video_out_M.write(dst_M)
    video_out_W.write(dst_W)
    
print("\nDone")
  
video_out_M.release()
video_out_W.release()