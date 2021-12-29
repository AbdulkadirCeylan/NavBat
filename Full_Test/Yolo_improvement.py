#!/usr/bin/env python
import gym
import rospy
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes, ObjectCount
from std_srvs.srv import *
import time
import math
import sim as vrep
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import random

class Tracking():

    def __init__(self):
        rospy.init_node("my_node",anonymous=True)
        rospy.Subscriber('darknet_ros/bounding_boxes',BoundingBoxes,self.box_callback)
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.done = False


    def box_callback(self,data):
        for box in data.bounding_boxes:
            self.x_min = box.xmin 
            self.x_max = box.xmax 
            self.y_max = box.ymax
            self.y_min = box.ymin 
            self.done = True


