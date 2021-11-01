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
import cv2
import imutils
import random


#register the training environment in the gym as an available one
reg = register(
    id='Quad_target_trying-v0',
    entry_point='env_target_try:QuadCopterEnv',
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

        self.err_code,self.target_handle_target = vrep.simxGetObjectHandle(self.clientID_aux,"Target",vrep.simx_opmode_blocking) # Target
        self.err_code,self.target_handle = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter_target",vrep.simx_opmode_blocking) # Target
        self.err_code,self.target_handle_1 = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter",vrep.simx_opmode_blocking)      # Drone itself
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_blocking) # Target pos
        self.err_code,self.local_pos_drone = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_blocking) # Drone pos
        self.err_code, self.eulers = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Drone Orientation
        self.best_distance = 20
        self.target_location = Point(0,0,0)
        # stablishes connection with simulator
        self.action_space = spaces.Discrete(3) #Forward,Left,Right,Back
        self.reward_range = (-np.inf, np.inf)
        self.seed()

    def calculate_step_dist(self):
        self.err_code,self.target_location = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_blocking) # Target pos
        self.err_code,self.local_pos_drone = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_blocking) # Drone pos

        x_ax = (self.target_location[0] - self.local_pos_drone[0])
        y_ax = (self.target_location[1] - self.local_pos_drone[1])
        remained_distance =  math.sqrt(pow(x_ax,2)+pow(y_ax,2))
        return remained_distance

    def calculate_target_distance(self):
        self.err_code,self.target_location = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_target,-1,vrep.simx_opmode_blocking) # Target pos
        self.err_code,self.local_pos_drone = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_blocking) # Drone pos
        self.err_code, self.local_angle = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Drone Orientation

        x_ax = (self.target_location[0] - self.local_pos[0])
        y_ax = (self.target_location[1] - self.local_pos[1])
        remained_distance =  math.sqrt(pow(x_ax,2)+pow(y_ax,2))
        theta = 0
        if y_ax != 0:
            theta = math.atan(x_ax/y_ax)

        theta = theta *100
        theta = round(theta)
        remained_distance = round(remained_distance*10)
        w = self.local_angle[2]*100
        w = round(w)
        print(theta,w)
        return remained_distance,theta,w

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

   
    def reset(self):
        self.best_distance = 20
        y = random.randint(-7,5)
        y_target = random.randint(-7,5)
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_target,-1,[12,y_target,1],vrep.simx_opmode_streaming)  # Problem is opmode_streaming
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_1,-1,[0,y,1],vrep.simx_opmode_streaming)  # Problem is opmode_streaming
        status_one = vrep.simxStopSimulation(self.clientID_aux, vrep.simx_opmode_oneshot_wait)
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle,-1,[0,y,1],vrep.simx_opmode_streaming)  # Set Object Position
        time.sleep(1)
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_1,-1,[0,y,1],vrep.simx_opmode_oneshot)  # Problem is opmode_streaming
        status = vrep.simxStartSimulation(self.clientID_aux, vrep.simx_opmode_oneshot_wait)
        time.sleep(1)
        data_pose, data_imu = self.take_observation()
        fresh_distance,theta,w = self.calculate_target_distance()
        observation = [theta,w]
        return observation


    def step(self, action):

        if action != 2:
            self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
            self.err_code,self.local_yaw = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)
            w = self.local_yaw[2]
            if (action == 0):
                degree = -45*math.pi/180
                err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,w+degree],vrep.simx_opmode_streaming)  # Rigth

            #if (action == 1): 
                #degree = 0*math.pi/180
                #err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,w+degree],vrep.simx_opmode_streaming)  # Left

            if (action == 1): 
                degree = 45*math.pi/180
                err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,w+degree],vrep.simx_opmode_streaming)  # Left

            x = w+degree
            x_deg = abs(x * 180/math.pi)
            desired_x = abs(self.local_angles[2]*180/math.pi)
            sum_360 = x_deg + desired_x
        
            while not (abs(self.local_angles[2])-0.1 < abs(x) and  abs(x) < abs(self.local_angles[2])+0.1) and not ((sum_360)+5 > 360 and (sum_360)-5 < 360):
                self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)
                desired_x = abs(self.local_angles[2]*180/math.pi)
                sum_360 = x_deg + desired_x

            
        self.err_code,self.local_yaw = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)
        w = self.local_yaw[2]
        loc_x = self.local_pos[0]    
        loc_y = self.local_pos[1]
        move_as_x = math.cos(w)*0.4
        move_as_y = math.sin(w)*0.4
        self.distance = 1

        if (action == 2):
            err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle,-1,[loc_x+move_as_x,loc_y +move_as_y,1],vrep.simx_opmode_streaming)
            while self.distance > 0.15:
                self.distance =  self.calculate_step_dist()

        dst,theta,w = self.calculate_target_distance()
        data_pose, data_imu = self.take_observation()
        reward,done = self.process_data(data_pose, data_imu,dst,self.eulers) 

        state = [theta,w]
        return state, reward, done, {}


    def take_observation (self):
        err_code, self.eulers = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        err_code, self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        data_pose = None
        while data_pose is None:
            data_pose = self.local_pos

        data_imu = None
        while data_imu is None:
            data_imu = self.eulers    
        
        return data_pose,data_imu

    def process_data(self, data_position, data_imu,dst,eulerr):
        pitch = eulerr[0]*180/math.pi
        roll = eulerr[1]*180/math.pi
        pitch_bad = not(-15 < pitch < 15)
        roll_bad = not(-15 < roll < 15)
        done = False
        reward = 0
        dst = dst/10
        if data_position[0] > 13 or data_position[0]<-3 or data_position[1] <-7 or data_position[1] > 7:
            print("Drone is out of the area") 
            done = True

        if pitch_bad or roll_bad:
            reward -= 100
            done = True
            print("Crashed")

        if dst < self.best_distance:

            self.best_distance = dst
            reward += 2

        if dst > self.best_distance:
            reward -= 2

        if dst < 1.2:
            reward += 500
            done = True
            print("Finished Succesfully")

        return reward,done