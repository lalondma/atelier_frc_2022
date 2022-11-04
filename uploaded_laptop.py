#!/usr/bin/env python3

import time
import numpy as np
import pupil_apriltags as apriltag
import math
import cv2

IMGW = 640 #1280
IMGH = 480 #960
HIRES=False
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
    objectPoints = np.array([[[-80, -80, 0], \
    [80, -80, 0], \
    [80, 80, 0], \
    [-80, 80, 0]]], \
    dtype=np.float32)
    
    rVec = None
    tVec = None
    
    # loop forever
    flag = True
    iterate = False
    
    # Params camera avec image format 640x480
    distCoeff = np.array([-0.159,  0.526, 0.0098,  -0.0014,  -0.888])
    focal_length = 600
    center = [320, 240]
    camMatrix = np.array(
        [[680, 0, 301],
        [0, 680, 237],
        [0, 0, 1]], dtype = "double")
    if HIRES:
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
        # Détection du tag
        results = detector.detect(gray)
        
        # loop over the AprilTag detection results
        for r in results:
           # extract the bounding box (x, y)-coordinates for the AprilTag
           # and convert each of the (x, y)-coordinate pairs to integers
           (ptA, ptB, ptC, ptD) = r.corners
           image = imagergb.copy()
           ptB = (int(ptB[0]), int(ptB[1]))
           ptC = (int(ptC[0]), int(ptC[1]))
           ptD = (int(ptD[0]), int(ptD[1]))
           ptA = (int(ptA[0]), int(ptA[1]))
           # draw the bounding box of the AprilTag detection
           cv2.line(image, ptA, ptB, (0, 255, 0), 2)
           cv2.line(image, ptB, ptC, (0, 255, 0), 2)
           cv2.line(image, ptC, ptD, (0, 255, 0), 2)
           cv2.line(image, ptD, ptA, (0, 255, 0), 2)
           # draw the center (x, y)-coordinates of the AprilTag
           (cX, cY) = (int(r.center[0]), int(r.center[1]))
           cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
           tagFamily = r.tag_family.decode("utf-8")
           #print("[INFO] tag family: {}".format(tagFamily))
           # Affichage
           image = cv2.resize(image, (320,240))
           cv2.imshow("Apriltag detection", image)
           
           # April Tags have sub-pixel accuracy already
           imagePoints = r.corners.reshape(1,4,2)
           
           # Calcul de pose de camera 
#           good, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints, \
#             camMatrix, distCoeff, rVec, tVec, iterate, cv2.SOLVEPNP_EPNP) #ITERATIVE)
           _,rVec, tVec= cv2.solvePnP(objectPoints, imagePoints, \
               camMatrix, distCoeff, cv2.SOLVEPNP_ITERATIVE)
           good=True
           if good:
              # Detection
              iterate = True
              ZYX,jac = cv2.Rodrigues(rVec)
              
              totalrotmax=np.array([[ZYX[0,0],ZYX[0,1],ZYX[0,2],tVec[0]],[ZYX[1,0],ZYX[1,1],ZYX[1,2],tVec[1]],[ZYX[2,0],ZYX[2,1],ZYX[2,2],tVec[2]],[0,0,0,1]], dtype=np.float64    )
              WtoC=np.mat(totalrotmax)

              inverserotmax=np.linalg.inv(totalrotmax)
              f=inverserotmax
              if not counter%15:  # 2x par seconde
                 # Division par 1000 pour avoir des metres                 
                 x=f[0,3]/1000;   
                 y=f[1,3]/1000
                 z=f[2,3]/1000
                 print("xyz: %.1f, %.1f, %.1f" % (x,y,z))
                 print("distance: %.1f" % math.sqrt(x*x+y*y+z*z))

            # Can not be 'behind' barcode, or too far away
           if tVec[2][0] < 0 or tVec[2][0] > 10000:
                rVec = None
                tVec = None
                iterate = False
                
           
        counter += 1
        cv2.waitKey(1)
        if (time.time()-start_time) > freq:
          print("FPS: ", counter/(time.time()-start_time))
          counter = 0
          start_time = time.time()   
