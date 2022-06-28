
#This program takes a video file and removes the camera distortion based on the
#camera calibration parameters.

#This program first loads the calibration data. The file then loops through each frame from the input video,
#undistorts the frame and then saves the resulting frame into the output video.
#It should be noted that the audio from the input file is not transfered to the
#output file.

#sources:
#https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#https://www.theeminentcodfish.com/gopro-calibration/

import numpy as np
from glob import glob
from time import perf_counter
import cv2
import os
import sys

gopro_videos = sorted(glob("videos/gopro/MSK*")) # "videos/gopro/MSK" als je calibratievideos niet wilt omzetten
calibration_files = sorted(glob("*.npz"))

distCoeff = []
intrinsic_matrix = []

for calibration_file in calibration_files:
    
    print ('Loading calibration file ' + calibration_file)
        
    npz_calib_file = np.load(calibration_file)
    
    distCoeff.append(npz_calib_file['distCoeff'])
    intrinsic_matrix.append(npz_calib_file['intrinsic_matrix'])
    
    npz_calib_file.close()
    
    print('Finished loading file')
    print(' ')
    
    
#makes seperate directory for the calibration images
old_directory = os.getcwd()
subdirectory = 'videos\gopro_calibrated_videos'
final_directory = os.path.join(old_directory, subdirectory )
if not os.path.exists(final_directory):
    os.makedirs(final_directory)
    
#print(os.getcwd())
    
for filename in gopro_videos:
    crop = 0.5
    
    # remove pathname
    base=os.path.basename(filename)
    # get filename without extension
    file = os.path.splitext(base)[0]
    
    print('Starting to undistort video ' + file)
    aantal = len(calibration_files)
    print(str(aantal)+ " files will be created because there are " + str(aantal) + " calibration files")
    
    #Opens the video import and sets parameters
    video = cv2.VideoCapture(filename)
    #Checks to see if a the video was properly imported
    status = video.isOpened()
    
    if status == True:

        #FPS = video.get(cv2.CAP_PROP_FPS)
        FPS = 60.0
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (int(width), int(height))
        print("FPS = ", FPS, "\nheight = ", height, " pixels\nwidth = ", width, " pixels")
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_lapse = (1/FPS)*1000
    
        #change current directory to the directory /videos/gopro_calibrated_videos
        #os.chdir(final_directory)
    
        #Initializes the export video file
        codec = cv2.VideoWriter_fourcc(*'XVID')
        video_out_M = cv2.VideoWriter(file +'_undistored_with_M.avi', codec, FPS, size, 1) # 1 signifies isColor == True
        video_out_W = cv2.VideoWriter(file +'_undistored_with_W.avi', codec, FPS, size, 1) 
    
        #Initializes the frame counter
        current_frame = 0
        t1_start = perf_counter()
        
        newMat_M, ROI_M = cv2.getOptimalNewCameraMatrix(intrinsic_matrix[0], distCoeff[0], size, alpha = crop, centerPrincipalPoint = 1)
        mapx_M, mapy_M = cv2.initUndistortRectifyMap(intrinsic_matrix[0], distCoeff[0], None, newMat_M, size, m1type = cv2.CV_32FC1)
        
        newMat_W, ROI_W = cv2.getOptimalNewCameraMatrix(intrinsic_matrix[1], distCoeff[1], size, alpha = crop, centerPrincipalPoint = 1)
        mapx_W, mapy_W = cv2.initUndistortRectifyMap(intrinsic_matrix[1], distCoeff[1], None, newMat_W, size, m1type = cv2.CV_32FC1)
    
        print("\nUndistorting frames...")
    
        x = 0.05
        # example length of loading bar
        for i in range(int(1/x)):
            print("-", end = "")
        print("")
        
        # use matrix on al frames of the video
        while current_frame < total_frames:
            # loading bar
            if (current_frame >= x*total_frames):
                 print('\u2588', end="")
                 x += 0.05
            
            success, image = video.read()
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    
            dst_M = cv2.remap(image, mapx_M, mapy_M, cv2.INTER_LINEAR)
            dst_W = cv2.remap(image, mapx_W, mapy_W, cv2.INTER_LINEAR)
            
            #other way to undistort
            dst_M = cv2.undistort(image, intrinsic_matrix[0], distCoeff[0], None)
            dst_W = cv2.undistort(image, intrinsic_matrix[1], distCoeff[1], None)
        
            video_out_M.write(dst_M)
            video_out_W.write(dst_W)
        
        cv2.imshow("dst_M", dst_M)
        
        video.release()
        video_out_M.release()
        video_out_W.release()
        
        t1_stop = perf_counter()
        duration = (t1_stop - t1_start) #nanoseconds
        duration = duration/(60) #minutes
        
        #change back to general directonry to load the next video
        #os.chdir(old_directory)
    
        print(' ')
        print('Finished undistorting the video ' + file)
        print('This video took: ' + str(duration) + ' minutes\n')
    else:
        print("Error: Video failed to load")
        #sys.exit()


