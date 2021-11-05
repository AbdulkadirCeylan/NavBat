#!/usr/bin/env python
from os import PRIO_PGRP
import sim as vrep
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
import rospy
from darknet_ros_msgs.msg import BoundingBoxes, ObjectCount
import PRM
import DQN_trained_model


rospy.init_node("my_node",anonymous=True)

vrep.simxFinish(-1)
clientID_aux = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)


if clientID_aux!=-1:
    print ("Connected to remote API server")
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")

class Main:
    def __init__(self):  
        self.err_code,self.target_handle = vrep.simxGetObjectHandle(clientID_aux,"Quadcopter_target",vrep.simx_opmode_blocking)
        self.err_code,self.target_handle_1 = vrep.simxGetObjectHandle(clientID_aux,"Quadcopter",vrep.simx_opmode_blocking)
        rospy.Subscriber('darknet_ros/bounding_boxes',BoundingBoxes,self.box_callback)
        self.escape = False


    def PRM_action(self,rx_sampled,ry_sampled):
        target_reached = False
        step_num = 0
        done = False
        self.escape = False
        for k in range(len(rx_sampled)):
            if self.escape is False:   # if no human detected follow the PRM Path
                _, local_angles = vrep.simxGetObjectOrientation(clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
                _, local_pos = vrep.simxGetObjectPosition(clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)   # Get Object Orientation
                x_ax = local_pos[0]
                y_ax = local_pos[1]

                plt.plot(x_ax, y_ax, '-p', color='gray',
                    markersize=10, linewidth=4,
                    markerfacecolor='white',
                    markeredgecolor='gray',
                    markeredgewidth=2)
                plt.pause(0.00001)
                plt.title("Position of Drone")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.xlim(0,30)
                plt.ylim(0,30)
                #plt.show()
                _, theta = self.calculate_step_dist(rx_sampled[k],ry_sampled[k])
                x = theta
                _ = vrep.simxSetObjectOrientation(clientID_aux,self.target_handle,-1,[0,0,x],vrep.simx_opmode_streaming)  # Left
                x_deg = abs(x * 180/math.pi)
                desired_x = abs(local_angles[2]*180/math.pi)
                sum_360 = x_deg + desired_x
                t0 = time.time()
                while not (abs(local_angles[2])-0.1 < abs(x) and  abs(x) < abs(local_angles[2])+0.1) and not ((sum_360)+5 > 360 and (sum_360)-5 < 360):
                    _, local_angles = vrep.simxGetObjectOrientation(clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_oneshot_wait)
                    desired_x = abs(local_angles[2]*180/math.pi)
                    sum_360 = x_deg + desired_x
                    if(time.time()-t0) >4:
                        break
                _ = vrep.simxSetObjectPosition(clientID_aux,self.target_handle,-1,[rx_sampled[k],ry_sampled[k],1],vrep.simx_opmode_streaming)
                dst = 1
                t1 = time.time()
                while dst > 0.1:
                    dst,theta = self.calculate_step_dist(rx_sampled[k],ry_sampled[k])
                    if(time.time()-t1) >4:
                        break

            else: # Else run RL code to escape from human
                done = DQN_trained_model.main()  # Replace deneme_code with Trained RL Model
                print("RL CODE STARTS")
                break

            step_num = step_num + 1

            if step_num == len(rx_sampled):
                target_reached = True

        return target_reached,done

    def box_callback(self,data):
        for box in data.bounding_boxes:
            if box.Class == "person":
                self.escape = True # if a person seen enabled the RL Code



    def calculate_step_dist(self,target_location_x, target_location_y):
        err_code,local_pos_drone = vrep.simxGetObjectPosition(clientID_aux,self.target_handle_1,-1,vrep.simx_opmode_blocking) # Drone pos

        x_ax = (target_location_x - local_pos_drone[0])
        y_ax = (target_location_y - local_pos_drone[1])
        theta = 0

        if x_ax >0 and y_ax > 0:
            theta = math.atan(y_ax/x_ax)

        if x_ax <0 and y_ax >0:
            theta = math.atan(y_ax/x_ax)
            theta = theta - math.pi

        if x_ax <0 and y_ax <0:
            theta = math.atan(y_ax/x_ax)
            theta = theta - math.pi

        if x_ax >0 and y_ax <0:
            theta = math.atan(y_ax/x_ax)

        remained_distance =  math.sqrt(pow(x_ax,2)+pow(y_ax,2))

        return remained_distance,theta

    def main(self):
        target_reached = False
        done = True
        while target_reached is False:
            if done is True: # if RL code finished Start PRM Path again starting from your current location
                err_code, local_pos = vrep.simxGetObjectPosition(clientID_aux,self.target_handle,-1,vrep.simx_opmode_oneshot_wait)
                start_x = local_pos[0]
                start_y = local_pos[1]
                rx_sampled, ry_sampled = PRM.mainn(start_x,start_y)
                target_reached,done = self.PRM_action(rx_sampled,ry_sampled)

if __name__ == '__main__':
    cl = Main()
    cl.main()