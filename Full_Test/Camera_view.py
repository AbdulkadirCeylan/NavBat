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
from darknet_ros_msgs.msg import BoundingBoxes


vrep.simxFinish(-1)
clientID_aux = vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
if clientID_aux!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")

fake_img = cv2.imread('mushroom.jpeg')
img_pub = rospy.Publisher('/vrep_image', Image,queue_size=10)
temporary_state=np.zeros(10)
temporary_state_cntboxes=np.zeros(10)
def box_callback(data):
    print(data.bounding_boxes[1].xmin)
    for i in range(len(data.bounding_boxes)):
        x_center = (data.bounding_boxes[i].xmin + (data.bounding_boxes[i].xmax - data.bounding_boxes[i].xmin)/2)/5
        y_center = data.bounding_boxes[i].ymin + (data.bounding_boxes[i].ymax - data.bounding_boxes[i].ymin)/2
        area = (data.bounding_boxes[i].xmax-data.bounding_boxes[i].xmin) * (data.bounding_boxes[i].ymax-data.bounding_boxes[i].ymin)
        area = round(area/1000)
        temporary_state[i]=area
        temporary_state_cntboxes[i]=x_center
rospy.Subscriber('darknet_ros/bounding_boxes',BoundingBoxes,box_callback)
rospy.init_node('image', anonymous=True)
rate = rospy.Rate(60.0)
res,v0=vrep.simxGetObjectHandle(clientID_aux,'Vision_sensor',vrep.simx_opmode_oneshot_wait) 
time.sleep(1)
err, resolution, image = vrep.simxGetVisionSensorImage(clientID_aux, v0, 0, vrep.simx_opmode_oneshot_wait)
t = time.time()
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
        rate.sleep()
    else:

        img = np.array(fake_img, dtype = np.uint8)
        msg_frame = CvBridge().cv2_to_imgmsg(img,"bgr8")
        img_pub.publish(msg_frame)
        vrep.simxFinish(-1)
        clientID_aux = vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
