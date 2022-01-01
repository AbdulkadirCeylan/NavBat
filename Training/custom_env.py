#!/usr/bin/env python
import gym
import rospy
import numpy as np
from gym import utils, spaces
from darknet_ros_msgs.msg import BoundingBoxes, ObjectCount
from gym.utils import seeding
from gym.envs.registration import register
from std_srvs.srv import *
import time
import math
import sim as vrep
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import random
from sklearn import preprocessing
import cv2
from cv_bridge import CvBridge, CvBridgeError
import imutils

#register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='custom_env:QuadCopterEnv',
    )

class QuadCopterEnv(gym.Env):

    def __init__(self):

        vrep.simxFinish(-1)
        self.clientID_aux = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

        if self.clientID_aux!=-1:
            print ("Connected to remote API server")
        else:
            print("Not connected to remote API server")
            sys.exit("Could not connect")

        rospy.init_node("custom_env",anonymous=True)
        rospy.Subscriber('darknet_ros/bounding_boxes',BoundingBoxes,self.box_callback)
        res,self.v0=vrep.simxGetObjectHandle(self.clientID_aux,'Vision_sensor',vrep.simx_opmode_oneshot_wait) 
        self.err_code,self.target_handle = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter_target",vrep.simx_opmode_blocking)
        self.err_code,self.target_handle_1 = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter",vrep.simx_opmode_blocking)
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_blocking)
        self.err_code,self.local_pos_drone = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_blocking)
        self.err_code, self.eulers = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        self.area = 0
        self.x_center = 0
        self.target_location = Point(0,0,0)
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box(low=0,high=50,shape=(1,20))
        self.observation_space = spaces.Discrete(1)
        self.reward_range = (-np.inf, np.inf)
        self.temporary_state=np.zeros(shape=(1,20))
        self.previous_area=0
        self.previous_x=0
        self.seed()
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.prev_x1 = 0
        self.prev_x2 = 0
        self.sub_img = 0

    def calculate_target_distance(self,target_location_x,target_location_y):
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)
        x_ax = (target_location_x - self.local_pos[0])
        y_ax = (target_location_y - self.local_pos[1])
        remained_distance =  math.sqrt(pow(x_ax,2)+pow(y_ax,2))
        return remained_distance


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def improvedYolo(self):
        err, resolution, Vrep_camera_image = vrep.simxGetVisionSensorImage(self.clientID_aux, self.v0, 0, vrep.simx_opmode_oneshot_wait)
        img = np.array(Vrep_camera_image, dtype = np.uint8)
        img.resize([resolution[0],resolution[0],3])
        Rotated1_image = imutils.rotate(img, angle=180)
        Vrep_camera_image = cv2.flip(Rotated1_image,1)
        
        x1 = self.x_min
        y1 = self.y_max
        x2 = self.x_max
        y2 = self.y_min
        if not (self.prev_x1==x1 and self.prev_x2 == x2):
            ### Draw bounding box on original image
            ### Sub-part of Detected image
            detected_greyscale = cv2.cvtColor(Vrep_camera_image, cv2.COLOR_BGR2GRAY)
            self.sub_img = detected_greyscale[y2:y1,x1:x2]

        #### Correlation ####
        #cor = signal.correlate2d (detected_greyscale,sub_img)
        
        method = eval('cv2.TM_CCOEFF_NORMED')
        v_rep_camera = cv2.cvtColor(Vrep_camera_image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(self.sub_img,v_rep_camera,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        h, w = self.sub_img.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(Vrep_camera_image,top_left, bottom_right, 255, 2)

        #print(cor.shape)
        #print(x1,x2,y1,y2)
        cv2.imshow('frame', Vrep_camera_image)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("exit")
        self.prev_x1 = x1
        self.prev_x2 = x2


    def box_callback(self,data):
        #self.temporary_state=np.zeros(shape=(1,20))
        #print(data.bounding_boxes[1].xmin)

        for i in range(len(data.bounding_boxes)):
            self.x_min = data.bounding_boxes[i].xmin 
            self.x_max = data.bounding_boxes[i].xmax 
            self.y_max = data.bounding_boxes[i].ymax
            self.y_min = data.bounding_boxes[i].ymin  
            #self.done = True
            # self.x_center = (data.bounding_boxes[i].xmin + (data.bounding_boxes[i].xmax - data.bounding_boxes[i].xmin)/2)
            # self.y_center = data.bounding_boxes[i].ymin + (data.bounding_boxes[i].ymax - data.bounding_boxes[i].ymin)/2
            # self.area = (data.bounding_boxes[i].xmax-data.bounding_boxes[i].xmin) * (data.bounding_boxes[i].ymax-data.bounding_boxes[i].ymin)
            # self.area = round(self.area/250)
            # if self.x_center > 128:
            #     self.x_center = (self.x_center - 128)/5
            # else:
            #     self.x_center = (128 - self.x_center)/5
                
            #self.temporary_state[0,i]=self.area
            #self.temporary_state[0,i+10]=self.x_center

            
    def reset(self):
        self.area = 0
        y = random.randint(-2,2)
        x = random.randint(-2,2)
        yaw = random.random()*3
        #y_target = random.randint(-2,2)
        #err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_target,-1,[12,y_target,1],vrep.simx_opmode_streaming)  # Problem is opmode_streaming
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_1,-1,[x,y,1],vrep.simx_opmode_streaming)
        err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,[0,0,yaw],vrep.simx_opmode_streaming)
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle,-1,[x,y,1],vrep.simx_opmode_streaming)  
        err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,yaw],vrep.simx_opmode_streaming)  
        status_one = vrep.simxStopSimulation(self.clientID_aux, vrep.simx_opmode_oneshot_wait)
        time.sleep(1)
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_1,-1,[x,y,1],vrep.simx_opmode_oneshot) 
        err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,[0,0,yaw],vrep.simx_opmode_oneshot) 
        status = vrep.simxStartSimulation(self.clientID_aux, vrep.simx_opmode_oneshot_wait)
        time.sleep(1)
        #data_pose, data_imu = self.take_observation()
        #fresh_distance,theta =  self.calculate_target_distance()
        observation = self.x_center
        return observation

    def step(self, action):
        self.improvedYolo()
        self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        self.err_code,self.local_yaw = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)
        w = self.local_yaw[2]
        if (action == 0):
            degree = -30*math.pi/180
            err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,w+degree],vrep.simx_opmode_streaming)  # Rigth

        if (action == 1): 
            degree = 0*math.pi/180
            err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,w+degree],vrep.simx_opmode_streaming)  # Left

        if (action == 2): 
            degree = 30*math.pi/180
            err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,w+degree],vrep.simx_opmode_streaming)  # Left

        x = w+degree
        x_deg = abs(x * 180/math.pi)
        desired_x = abs(self.local_angles[2]*180/math.pi)
        sum_360 = x_deg + desired_x
        t0 = time.time()
        while not (abs(self.local_angles[2])-0.1 < abs(x) and  abs(x) < abs(self.local_angles[2])+0.1) and not ((sum_360)+5 > 360 and (sum_360)-5 < 360):
            self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)
            desired_x = abs(self.local_angles[2]*180/math.pi)
            sum_360 = x_deg + desired_x
            if(time.time()-t0) >10:
                break
            
        self.err_code,self.local_yaw = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)
        w = self.local_yaw[2]
        loc_x = self.local_pos[0]    
        loc_y = self.local_pos[1]
        move_as_x = math.cos(w)*0.3
        move_as_y = math.sin(w)*0.3
        self.distance = 1
        t1 = time.time()
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle,-1,[loc_x+move_as_x,loc_y +move_as_y,1],vrep.simx_opmode_streaming)
        while self.distance > 0.15:
            self.distance =  self.calculate_target_distance(loc_x+move_as_x,loc_y +move_as_y)
            if(time.time()-t1) >10:
                break

        data_pose, data_imu = self.take_observation()
        reward,done = self.process_data(data_pose, data_imu,self.distance) 

        if(self.previous_area==self.area and self.previous_x==self.x_center):
            #self.temporary_state=np.ones(shape=(1,20))
            self.area = 0
            self.x_center = 128/5
        
        state = self.x_center
        self.previous_area=self.area
        self.previous_x=self.x_center
        return state, reward, done, {}

    def take_observation (self):
        err_code, self.eulers = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        data_pose = None
        while data_pose is None:
            data_pose = self.local_pos

        data_imu = None
        while data_imu is None:
            data_imu = self.eulers    
        
        return data_pose,data_imu

    def process_data(self, data_position, data_imu,dst):
        pitch = data_imu[0]*180/math.pi
        roll = data_imu[1]*180/math.pi
        pitch_bad = not(-10 < pitch < 10)
        roll_bad = not(-10 < roll < 10)
        done = False
        reward = 0

        if pitch_bad or roll_bad:
            done = True
            print("Falling")

        if data_position[0] > 13 or data_position[0] <-13.5 or data_position[1] > 10 or data_position[1] < -17:
            done = True
            print("Out of area")

        #if self.area[0,0:9].any() > 12:
            #reward -= 5
            #done = True
            #print("Crashed with Human")
            
        if self.x_center <= self.previous_x:
            reward += 10
        else:
            reward -=5 
        print(reward)
        return reward,done
