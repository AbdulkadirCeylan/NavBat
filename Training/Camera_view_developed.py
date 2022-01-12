#!/usr/bin/env python
import sys
import sim as vrep
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps
import imutils
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from Yolo_improvement import Tracking
import cv2
from scipy import signal
from std_msgs.msg import Float64

vrep.simxFinish(-1)
clientID_aux = vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
if clientID_aux!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")


color = (255, 0, 0)
tracking = Tracking()
fake_img = cv2.imread('mushroom.jpeg')
img_pub = rospy.Publisher('/vrep_image', Image,queue_size=10)
x_coor_pub = rospy.Publisher('/box_x_coor', Float64, queue_size=10)
box_area_pub = rospy.Publisher('/box_area', Float64, queue_size=10)
rate = rospy.Rate(20.0)
res,v0=vrep.simxGetObjectHandle(clientID_aux,'Vision_sensor',vrep.simx_opmode_oneshot_wait) 
time.sleep(1)
err, resolution, image = vrep.simxGetVisionSensorImage(clientID_aux, v0, 0, vrep.simx_opmode_oneshot_wait)
t = time.time()
prev_x1 = 0
prev_x2 = 0
detected_greyscale = 0
sub_img = 0
x_center = 0
box_history = np.zeros(15)
k = 0
area = 0
while True:
    #getting image
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID_aux, v0, 0, vrep.simx_opmode_oneshot_wait)
    if err == vrep.simx_return_ok:

        img = np.array(image, dtype = np.uint8)
        img.resize([resolution[0],resolution[0],3])
        Rotated1_image = imutils.rotate(img, angle=180)
        Rotated2_image = cv2.flip(Rotated1_image,1)
        msg_frame = CvBridge().cv2_to_imgmsg(Rotated2_image,"bgr8")
        img_pub.publish(msg_frame)
        if k >= 15:
            k = 0

        x1 = tracking.x_min
        y1 = tracking.y_max
        x2 = tracking.x_max
        y2 = tracking.y_min
        box_history[k] = x1
        if not (prev_x1==x1 and prev_x2 == x2):
            ### Draw bounding box on original image
            ### Sub-part of Detected image
            detected_greyscale = cv2.cvtColor(Rotated2_image, cv2.COLOR_BGR2GRAY)
            sub_img = detected_greyscale[y2:y1,x1:x2]

        result = all(elem == box_history[0] for elem in box_history)
        if result is True: 
            detected_greyscale = cv2.cvtColor(Rotated2_image, cv2.COLOR_BGR2GRAY)
            detected_greyscale[:,:] = 255
            sub_img= detected_greyscale
            
        
        
        method = eval('cv2.TM_CCOEFF_NORMED')
        v_rep_camera = cv2.cvtColor(Rotated2_image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(sub_img,v_rep_camera,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        h, w = sub_img.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(Rotated2_image,top_left, bottom_right, 255, 2)
        x_center = top_left[0]+(bottom_right[0]-top_left[0])/2
        area = h*w
        x_coor_pub.publish(x_center)
        box_area_pub.publish(area)
        cv2.imshow('frame', Rotated2_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_x1 = x1
        prev_x2 = x2
        k = k +1
    else:

        img = np.array(fake_img, dtype = np.uint8)
        msg_frame = CvBridge().cv2_to_imgmsg(img,"bgr8")
        img_pub.publish(msg_frame)
        x_coor_pub.publish(x_center)
        box_area_pub.publish(area)
        vrep.simxFinish(-1)
        clientID_aux = vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

