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
#register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v2',
    entry_point='custom_env_trained_model:QuadCopterEnv',
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

        #rospy.init_node("custom_env",anonymous=True)
        rospy.Subscriber('darknet_ros/bounding_boxes',BoundingBoxes,self.box_callback)
        #self.err_code,self.target_handle_target = vrep.simxGetObjectHandle(self.clientID_aux,"Target",vrep.simx_opmode_blocking) # Target
        self.err_code,self.target_handle = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter_target",vrep.simx_opmode_blocking)
        self.err_code,self.target_handle_1 = vrep.simxGetObjectHandle(self.clientID_aux,"Quadcopter",vrep.simx_opmode_blocking)
        self.err_code,self.local_pos = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_blocking)
        self.err_code,self.local_pos_drone = vrep.simxGetObjectPosition(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_blocking)
        self.err_code, self.eulers = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
        res,data_u1=vrep.simxGetStringSignal(self.clientID_aux,'myPtsu1',vrep.simx_opmode_streaming)
        res,data_u2=vrep.simxGetStringSignal(self.clientID_aux,'myPtsu2',vrep.simx_opmode_streaming)
        self.best_distance = 20
        self.area = 0
        self.x_center = 0
        self.previous_area = 0
        self.target_location = Point(0,0,0)
        self.prev_area = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(1)
        self.reward_range = (-np.inf, np.inf)
        self.seed()

    def readLidarData(self):
        res,data_u1=vrep.simxGetStringSignal(self.clientID_aux,'myPtsu1',vrep.simx_opmode_buffer)
        res,data_u2=vrep.simxGetStringSignal(self.clientID_aux,'myPtsu2',vrep.simx_opmode_buffer)
        data_u1=vrep.simxUnpackFloats(data_u1)
        data_u2=vrep.simxUnpackFloats(data_u2)
        data_u1=np.array(data_u1)
        data_u2=np.array(data_u2)
        
        if len(data_u1>1):
            laser_beam_u1=np.zeros(30)
            laser_beam_u2=np.zeros(30)
            for k in range(30):
                laser_beam_u1[k]=data_u1[5+4*k]
                laser_beam_u2[k]=data_u2[5+4*k]    
            laser_beam_u1 = np.insert(laser_beam_u1,30,laser_beam_u2)
            laser_beam_u1[laser_beam_u1<0.01]=2000
            r_min=laser_beam_u1[laser_beam_u1<1.5]    
            b_angles = np.where(laser_beam_u1<1.5) 
            if not len(r_min)==0:
                r=(r_min[0]+r_min[len(r_min)-1])/2
                angles = (b_angles[0][0]+b_angles[0][len(b_angles)-1])/2
            else:
                r = 2000
                angles = 180
        return r,angles

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
        
        for box in data.bounding_boxes:
            self.x_center = (box.xmin + (box.xmax - box.xmin)/2)/5
            self.y_center = box.ymin + (box.ymax - box.ymin)/2
            self.area = (box.xmax-box.xmin) * (box.ymax-box.ymin)
            self.area = round(self.area/1000)
            #if self.x_center > 128:
                #self.box_coor = (self.x_center - 128)/5
            #else:
                #self.box_coor = (128 - self.x_center)/5
            

    def reset(self):
        #self.area = 0
        #self.best_distance = 20
        #y = random.randint(-6,-5)
        #x = random.randint(-3,-2)
        #yaw = random.random()*3
        #y_target = random.randint(-2,2)
        #err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_target,-1,[12,y_target,1],vrep.simx_opmode_streaming)  # Problem is opmode_streaming
        #err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_1,-1,[x,y,1],vrep.simx_opmode_streaming)
        #err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,[0,0,yaw],vrep.simx_opmode_streaming)
        #err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle,-1,[x,y,1],vrep.simx_opmode_streaming)  
        #err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,yaw],vrep.simx_opmode_streaming)  
        #status_one = vrep.simxStopSimulation(self.clientID_aux, vrep.simx_opmode_oneshot_wait)
        #time.sleep(1)
        #err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle_1,-1,[x,y,1],vrep.simx_opmode_oneshot) 
        #err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,[0,0,yaw],vrep.simx_opmode_oneshot) 
        #status = vrep.simxStartSimulation(self.clientID_aux, vrep.simx_opmode_oneshot_wait)
        #time.sleep(1)
        #data_pose, data_imu = self.take_observation()
        #fresh_distance,theta =  self.calculate_target_distance()
        observation = self.x_center
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
        while not (abs(self.local_angles[2])-0.05 < abs(x) and  abs(x) < abs(self.local_angles[2])+0.05) and not ((sum_360)+5 > 360 and (sum_360)-5 < 360):
            self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)
            desired_x = abs(self.local_angles[2]*180/math.pi)
            sum_360 = x_deg + desired_x
            if(time.time()-t0) >5:
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
        err_code = vrep.simxSetObjectPosition(self.clientID_aux,self.target_handle,-1,[loc_x+move_as_x,loc_y +move_as_y,1.25],vrep.simx_opmode_streaming)
        while self.distance > 0.05:
            self.distance =  self.calculate_target_distance(loc_x+move_as_x,loc_y +move_as_y)
            if(time.time()-t1) >5:
                break
        

        ####### Lidar Distance Check ######
        distance,bearing = self.readLidarData()
        while distance < 1.5:
            distance,bearing = self.readLidarData()
            print(distance,bearing)
            self.err_code, self.local_angles = vrep.simxGetObjectOrientation(self.clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)
            w = self.local_angles[2]
            degree = -bearing*math.pi/180
            err_code = vrep.simxSetObjectOrientation(self.clientID_aux,self.target_handle,-1,[0,0,w+0.3],vrep.simx_opmode_streaming)  # Left

        data_pose, data_imu = self.take_observation()
        reward,done = self.process_data(data_pose, data_imu,self.distance,self.eulers) 

        if self.prev_area == self.area:
            self.area = 0
            self.x_center = 0
        else:
            self.prev_area = self.area

        state = self.x_center
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

    def process_data(self, data_position, data_imu,dst,eulerr):
        reward = 0
        done = False
        return reward,done
