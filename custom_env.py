#!/usr/bin/env python
import gym
import rospy
import time
import numpy as np
import tf
import time
from gym import utils, spaces
from geometry_msgs.msg import Point, PoseStamped,TwistStamped
from darknet_ros_msgs.msg import BoundingBoxes, ObjectCount
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from mavros_msgs.msg import *   
from mavros_msgs.srv import *
from std_srvs.srv import *
import time
from gazebo_connections import GazeboConnection
from fcu_modes import fcuModes


#register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='custom_env:QuadCopterEnv',
    )


class QuadCopterEnv(gym.Env):

    def __init__(self):
        # We assume that a ROS node has already been created
        # before initialising the environment
        #Publishers
        self.sp_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
        rospy.Subscriber('mavros/state', State, self.stateCb)
        rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.posCb)
        rospy.Subscriber('mavros/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('darknet_ros/bounding_boxes',BoundingBoxes,self.box_callback)
        self.gazebo = GazeboConnection()
        self.modes = fcuModes()
        self.rate = rospy.Rate(20.0)
        self.state = State()
        self.local_velocity = TwistStamped()
        self.imu_data = Imu()
        self.pose_command = PositionTarget()
        self.pose_command.type_mask = int('010111111000', 2)
        self.ALT_SP = 1
        # update the setpoint message with the required altitude
        self.pose_command.position.z = self.ALT_SP
        self.local_pos = Point(0,0,1)
        # Step size for position update
        self.STEP_SIZE = 2.0
		# Fence. We will assume a square fence for now

        self.imu = Imu()
        self.running_step = 100

        # stablishes connection with simulator
        
        
        self.action_space = spaces.Discrete(3) #Forward,Left,Right
        self.reward_range = (-np.inf, np.inf)

        self.seed()


    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z
        self.pose_command.position.x = self.local_pos.x
        self.pose_command.position.y = self.local_pos.y

    # A function to initialize the random generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def imu_callback(self,msg):
        self.imu_data.orientation.x = msg.orientation.x
        self.imu_data.orientation.y = msg.orientation.y
        self.imu_data.orientation.z = msg.orientation.z


    def box_callback(self,data):
        self.vertical_coor = 0
        for box in data.bounding_boxes:

            self.vertical_coor = (box.xmin) + (box.xmax - box.xmin)/2
            self.area = (box.xmax-box.xmin) * (box.ymax-box.ymin)
        #print(self.vertical_coor)

    # Resets the state of the environment and returns an initial observation.
    def reset(self):
        if not self.state.armed:
            self.gazebo.reset_world()
        # 1st: resets the simulation to initial values
        else: 
            
            while self.state.MODE_PX4_LAND is False:

                print("Land mode")
                self.modes.setAutoLandMode()

            while self.state.armed:
                self.modes.setDisarm()

            self.gazebo.pause()
            time.sleep(1)
            self.gazebo.reset_world()
            time.sleep(1)
            self.gazebo.unpause()

        self.takeoff_sequence()

        data_pose, data_imu = self.take_observation()
        observation = [data_pose.x]
        
        return observation

    def step(self, action): 
        if (action == 0): #FORWARD
            self.local_velocity.twist.linear.x = 5 #1.0 m/s forward
            self.local_velocity.twist.linear.y = 0

        elif (action == 1): #LEFT
            self.local_velocity.twist.linear.x = 0 #1 m/s Left
            self.local_velocity.twist.linear.y = -3

        else:
            self.local_velocity.twist.linear.x = 0 #1 m/s Right
            self.local_velocity.twist.linear.y = +3

        
        self.vel_pub.publish(self.local_velocity)
        self.sp_pub.publish(self.pose_command)
        self.rate.sleep()

        data_pose, data_imu = self.take_observation()
        reward,done = self.process_data(data_pose, data_imu)  
        state = [data_pose.x]
        return state, reward, done, {}

    def take_observation (self):
        data_pose = None
        while data_pose is None:
            try:
                data_pose = self.local_pos

            except:
                rospy.loginfo("Current drone pose not ready yet, retrying for getting robot pose")

        data_imu = None
        while data_imu is None:
            try:
                data_imu = self.imu_data
            except:
                rospy.loginfo("Current drone imu not ready yet, retrying for getting robot imu")
        
        return data_pose,data_imu


    def takeoff_sequence(self):
    # Make sure the drone is armed
        while not self.state.armed:
            self.modes.setArm()
            self.rate.sleep()



        altitude_bad = not(-0.2 < self.local_pos.z < 0.2)

        while altitude_bad:
            altitude_bad = not(-0.2 < self.local_pos.z < 0.2)
            time.sleep(0.2) 
            print(self.local_pos.z)
        
        while self.local_pos.z <0.9:
            self.pose_command.position.z = 1
            self.modes.setOffboardMode()
            self.sp_pub.publish(self.pose_command)
            self.rate.sleep()

        
    def process_data(self, data_position, data_imu):

        done = False
        pitch_bad = not(-0.2 < data_imu.orientation.x < 0.2)
        roll_bad = not(-0.2 < data_imu.orientation.z < 0.2)
        altitude_bad = not(0.2 < self.local_pos.z < 2)
        print(data_imu.orientation.x,data_imu.orientation.y)
        reward = 0

        if pitch_bad or roll_bad or altitude_bad:
            rospy.loginfo ("(Drone flight status is wrong) >>> ("+str(pitch_bad)+","+str(roll_bad)+")")
            done = True
            reward = -200

        elif data_position.z < -2 or data_position.y > 4 or data_position.y < -4:
            rospy.loginfo ("Drone is out of the area")
            
            done = True
            reward = -200
        
        elif self.area >13000:
            done = True
            rospy.loginfo ("Crashed")
            reward = -200


        elif data_position.x > 20:
            done = True
            rospy.loginfo ("End of the episode")
            reward = 500
            
        return reward,done
