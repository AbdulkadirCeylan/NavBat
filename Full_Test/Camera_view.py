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
rate = rospy.Rate(60.0)
res,v0=vrep.simxGetObjectHandle(clientID_aux,'Vision_sensor',vrep.simx_opmode_oneshot_wait) 
time.sleep(1)
err, resolution, image = vrep.simxGetVisionSensorImage(clientID_aux, v0, 0, vrep.simx_opmode_oneshot_wait)
t = time.time()
plt.show()

while True:
    #getting image
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID_aux, v0, 0, vrep.simx_opmode_oneshot_wait)
    if err == vrep.simx_return_ok:
        x1 = tracking.x_min
        y1 = tracking.y_max
        x2 = tracking.x_max
        y2 = tracking.y_min
        img = np.array(image, dtype = np.uint8)
        img.resize([resolution[0],resolution[0],3])
        Rotated1_image = imutils.rotate(img, angle=180)
        Rotated2_image = cv2.flip(Rotated1_image,1)
        msg_frame = CvBridge().cv2_to_imgmsg(Rotated2_image,"bgr8")
        img_pub.publish(msg_frame)

        ### Draw bounding box on original image
        cv2.rectangle(Rotated2_image,(x1, y1), (x2, y2), color, 2)
        cv2.imshow('frame', Rotated2_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #rate.sleep()

        
    else:

        img = np.array(fake_img, dtype = np.uint8)
        msg_frame = CvBridge().cv2_to_imgmsg(img,"bgr8")
        img_pub.publish(msg_frame)
        vrep.simxFinish(-1)
        clientID_aux = vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
