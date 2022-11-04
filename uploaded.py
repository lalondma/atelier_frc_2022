#!/usr/bin/env python3

# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.

import json
import time
import sys
import numpy as np
#import pupil_apriltags as apriltag
import apriltag
import math
from networktables import NetworkTablesInstance
import os

import cv2

team = 3986
server = False
cameraConfigs = []
switchedCameraConfigs = []
cameras = []
imW,imH = 640,480 #320,240
server_addr = "169.254.160.218"

if __name__ == "__main__":

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClient(server_addr)
        ntinst.startDSClient()

    print("[INFO] detecting AprilTags...")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
#    detector = apriltag.Detector(families="tag36h11")   #pupil-apriltag
    
    camera=cv2.VideoCapture(0)
    if not camera.isOpened():
        exit(1)
        
    img = np.zeros((imH,imW,3),dtype=np.uint8)
    if len(cameras)>0:
        cvsink = CvSink("sink") #CameraServer.getInstance().getVideo()
        cvsink.setSource(cameras[0])
     
    objectPoints = np.array([[[-80, -80, 0], \
    [80, -80, 0], \
    [80, 80, 0], \
    [-80, 80, 0]]], \
    dtype=np.float32)

    
    rVec = None
    tVec = None
    iterate=False
    distCoeff = np.array([-0.5,3.3,0.03,-0.05,-5.7]) #np.zeros((5,1))
    focal_length=600
    center=[320,240]
    camMatrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype = "double"
        )        
    print("Connected to NetworkTables: {}".format(ntinst.isConnected()))
    
    sd = ntinst.getTable("SmartDashboard")
    sd.putString("Apriltag","on")
    
    while True:
        entry=ntinst.getTable("SmartDashboard").getEntry("detections")
        
        t,imagergb = camera.read()
        gray=cv2.cvtColor(imagergb, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)
        
        # loop over the AprilTag detection results
        x = 0
        y = 0
        z = 0
        distance = 0
        if results:
        
           r = results[0]   # Assume only one tag per image
           # extract the bounding box (x, y)-coordinates for the AprilTag
           # and convert each of the (x, y)-coordinate pairs to integers
           (ptA, ptB, ptC, ptD) = r.corners
           ptB = (int(ptB[0]), int(ptB[1]))
           ptC = (int(ptC[0]), int(ptC[1]))
           ptD = (int(ptD[0]), int(ptD[1]))
           ptA = (int(ptA[0]), int(ptA[1]))
           (cX, cY) = (int(r.center[0]), int(r.center[1]))
           tagFamily = r.tag_family.decode("utf-8")
           
           # April Tags have sub-pixel accuracy already
           imagePoints = r.corners.reshape(1,4,2)
        
           good, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints, \
             camMatrix, distCoeff, rVec, tVec, iterate, cv2.SOLVEPNP_ITERATIVE)

           if good:
              iterate = True
              ZYX,jac = cv2.Rodrigues(rVec)
              
              totalrotmax=np.array([[ZYX[0,0],ZYX[0,1],ZYX[0,2],tVec[0]],[ZYX[1,0],ZYX[1,1],ZYX[1,2],tVec[1]],[ZYX[2,0],ZYX[2,1],ZYX[2,2],tVec[2]],[0,0,0,1]], dtype=np.float64    )
              WtoC=np.mat(totalrotmax)

              inverserotmax=np.linalg.inv(totalrotmax)
              f=inverserotmax
              # Division par 1000 pour avoir des metres                 
              x = f[0,3]/1000;   
              y = f[1,3]/1000
              z = f[2,3]/1000
              #print("xyz: %.1f, %.1f, %.1f" % (x,y,z))
              distance = math.sqrt(x*x+y*y+z*z)


           sd.putNumber("tx", x)
           sd.putNumber("ty", y)
           sd.putNumber("tz", z)
           sd.putNumber("distance", distance)
           
           
           
