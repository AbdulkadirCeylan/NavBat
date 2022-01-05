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
from std_msgs.msg import Float64


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
        rospy.Subscriber('/box_x_coor',Float64,self.box_callback)
        rospy.Subscriber('/box_area',Float64,self.box_area)
        #self.err_code,self.target_handle_target = vrep.simxGetObjectHandle(self.clientID_aux,"Target",vrep.simx_opmode_blocking) # Target
        self.err_code,self.target_handle = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter_target",vrep.simx_opmode_blocking)
        self.err_code,self.target_handle_1 = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter",vrep.simx_opmode_blocking)
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_blocking)
        self.err_code,self.local_pos_drone = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_blocking)
        self.err_code, self.eulers = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        self.x_center = 0
        self.area = 0
        self.previous_area = 0
        self.previous_x = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0,high=50,shape=(1,2))
        self.reward_range = (-np.inf, np.inf)
        self.temporary_state=np.zeros(shape=(1,2))
        self.previous_x= 0
        self.seed()


    def calculate_target_distance(self,target_location_x,target_location_y):
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)
        x_ax = (target_location_x - self.local_pos[0])
        y_ax = (target_location_y - self.local_pos[1])
        remained_distance =  math.sqrt(pow(x_ax,2)+pow(y_ax,2))
        return remained_distance


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def box_callback(self,data):
        self.x_center = np.round(data.data/256,decimals=2)

    def box_area(self,data):
        self.area = np.round(data.data/26000,decimals=2)

    def reset(self):
        self.area = 0
        y = random.randint(-2,2)
        x = random.randint(-2,2)
        yaw = random.random()*3
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
   
        observation = self.temporary_state
        return observation

    def step(self, action):
        
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
        self.temporary_state[0,0]= self.area
        self.temporary_state[0,1] = self.x_center

        if self.previous_area==self.area and self.previous_x==self.x_center:
            self.temporary_state=np.zeros(shape=(1,2))


        state = self.temporary_state
        self.previous_area=self.area
        self.previous_x=self.x_center
        print(self.area,self.x_center)
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

        if self.area > self.previous_area:
            reward -= 2
            
        if self.x_center <= self.previous_x:
            reward += 5
        else:
            reward -=5 
        print("Reward: ",reward)
        return reward,done
