#!/usr/bin/env python3

import time
import numpy as np
import pupil_apriltags as apriltag
import math
import cv2

IMGW = 1280
IMGH = 960
CHECKERBOARD = (8,6)
if __name__ == "__main__":


    print("[INFO] detecting AprilTags...")
    #options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(families="tag36h11")
    
    # Ouvrir la camera
    camera=cv2.VideoCapture(0)
    if not camera.isOpened():
        print("No camera!")
        exit(1)
        
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMGW)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMGH)

    img = np.zeros((IMGH, IMGW, 3),dtype=np.uint8)
     
    # Paramètres du apriltag: carre noir de 16cmx16cm 
    # Coord x,y,z = 0,0,0 = centre du tag
    objectPoints = np.zeros((1, CHECKERBOARD[0]
                      * CHECKERBOARD[1],
                      3), np.float32)
# Petit board
    objectPoints[0, :, :2] = 36*np.mgrid[0:CHECKERBOARD[0],
                               0:CHECKERBOARD[1]].T.reshape(-1, 2)

    
    rVec = None
    tVec = None
    
    # loop forever
    flag = True
    iterate = False
    
    
    # Params camera avec image format 1280x960
    camMatrix = np.array(
        [[982, 0, 639],
        [0, 981, 333],
        [0, 0, 1]], dtype = "double")
        
    distCoeff = np.array([-0.138,  0.27, 0.000028,  -0.0003,  -0.22])
    
    # Calcul de fps
    start_time = time.time()
    counter = 0
    freq = 1

    while True:
        #  Capture d'une trame video
        t,imagergb = camera.read()
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(imagergb, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
                        gray, CHECKERBOARD,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
    
        # loop over the AprilTag detection results
        if ret:
           # extract the bounding box (x, y)-coordinates for the AprilTag
           # and convert each of the (x, y)-coordinate pairs to integers
           criteria = (cv2.TERM_CRITERIA_EPS +
               cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

           corners2 = cv2.cornerSubPix(
              gray, corners, CHECKERBOARD, (-1, -1), criteria)

           image = imagergb.copy()
           image = cv2.drawChessboardCorners(image,
                                              CHECKERBOARD,
                                              corners2, ret)
 
           image = cv2.resize(image, (320,240))
           cv2.imshow("Apriltag detection", image)
           
           # April Tags have sub-pixel accuracy already
           imagePoints = corners2.reshape(1,48,2)
           
           # Calcul de pose de camera 
#           good, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints, \
#             camMatrix, distCoeff, rVec, tVec, iterate, cv2.SOLVEPNP_EPNP) #ITERATIVE)
           _,rVec, tVec, _ = cv2.solvePnPRansac(objectPoints, corners2, \
               camMatrix, distCoeff) #ITERATIVE)
           good=True
           if good:
              # Detection
              iterate = True
              ZYX,jac = cv2.Rodrigues(rVec)
              
              totalrotmax=np.array([[ZYX[0,0],ZYX[0,1],ZYX[0,2],tVec[0]],[ZYX[1,0],ZYX[1,1],ZYX[1,2],tVec[1]],[ZYX[2,0],ZYX[2,1],ZYX[2,2],tVec[2]],[0,0,0,1]], dtype=np.float64    )

#The resulting array is the transformation matrix from world coordinates (centered on the target) to camera coordinates. (Centered on the camera) We need camera to world. That is just the inverse of that matrix.

              WtoC=np.mat(totalrotmax)

              inverserotmax=np.linalg.inv(totalrotmax)
              f=inverserotmax
              #print(inverserotmax)

#The inverserotmax is the 4x4 homogeneous transformation matrix for camera to world coordinates.

#The location of the camera in world coordinates is given by the elements of column 4, the translation vector. The rotation matrix is given by the 3x3 rotation submatrix.

#If you need the euler angles out of that rotation matrix, the Rodrigues function can also provide that, but we know that the rotation angles in the inverse transformation are just -1*rvec from the original matrix.
              if not counter%15:  # 2x par seconde
                 # Division par 1000 pour avoir des metres
                 
                 x=f[0,3]/1000;   
                 y=f[1,3]/1000
                 z=f[2,3]/1000
                 print("xyz: %.1f, %.1f, %.1f" % (x,y,z))
                 print("distance: %.1f" % math.sqrt(x*x+y*y+z*z))

            # Can not be 'behind' barcode, or too far away
           #if tVec[2][0] < 0 or tVec[2][0] > 10000:
            #    rVec = None
             #   tVec = None
              #  iterate = False
                
           
        counter += 1
        cv2.waitKey(1)
        if (time.time()-start_time) > freq:
          print("FPS: ", counter/(time.time()-start_time))
          counter = 0
          start_time = time.time()   
