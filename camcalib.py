# Import required modules:
import cv2
import sys
import numpy as np
import os
from time import sleep
# Define the dimensions of checkerboard
CHECKERBOARD = (8,6)
MIN_POINTS = 10
IMGW = 1280
IMGH = 960
#RECORD = True
 
# Stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Vector for the 3D points:
threedpoints = []
 
# Vector for 2D points:
twodpoints = [] 
 
# 3D points real world coordinates:
objectp3d = np.zeros((1, CHECKERBOARD[0]
                      * CHECKERBOARD[1],
                      3), np.float32)
# Feuille
objectp3d[0, :, :2] = 26.5*np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)
# Petit board
objectp3d[0, :, :2] = 36*np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


cap = cv2.VideoCapture(0)
FPS = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMGW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMGH)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
signal_capture=False    
while True:
        ret, img = cap.read()
        image = img        
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        # Find the chess board corners
        # if desired number of corners are
        # found in the image then ret = true:
        ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            corners2 = cv2.cornerSubPix(
              grayColor, corners, CHECKERBOARD, (-1, -1), criteria)

            if signal_capture:
                sys.stdout.write("\rCapture {}".format(len(threedpoints)))
                threedpoints.append(objectp3d)
                signal_capture = False
 
                # Refining pixel coordinates
                # for given 2d points.

                twodpoints.append(corners2)
                # When we have minimum number of data points, stop:
                if len(twodpoints) > MIN_POINTS:
                    cap.release()
                    cv2.destroyAllWindows()
                    break
 
            # Draw and display the corners:
            image = cv2.drawChessboardCorners(image,
                                              CHECKERBOARD,
                                              corners2, ret)
 
        cv2.imshow('img', image)
        
        
        # wait for ESC key to exit and terminate feed.
        k = cv2.waitKey(1)
        if k==32:
            signal_capture=True
        if k == 27:         
            cap.release()
            cv2.destroyAllWindows()
            break

            h, w = image.shape[:2] 


# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints):
print("Calculs...")
print(len(threedpoints))
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None) 
 
# Displaying required output
print(" Camera matrix:")
print(matrix)
 
print("\n Distortion coefficient:")
print(distortion)
 
mean_error = 0
for i in range(len(threedpoints)):
    imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(twodpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(threedpoints)) 
#print("\n Rotation Vectors:")
#print(r_vecs)
 
#print("\n Translation Vectors:")
#print(t_vecs)