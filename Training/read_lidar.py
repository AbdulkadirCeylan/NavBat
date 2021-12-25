#!/usr/bin/env python
import sim as vrep
import rospy
import numpy as np

vrep.simxFinish(-1)
clientID_aux = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

err_code,lidar_handle = vrep.simxGetObjectHandle(clientID_aux,"LaserScanner_2D",vrep.simx_opmode_blocking)
err_code,state,lidar_data,obj_handle,_ = vrep.simxReadProximitySensor(clientID_aux, lidar_handle, vrep.simx_opmode_streaming)

while True:
    #err_code,state,lidar_data,obj_handle,_ = vrep.simxReadProximitySensor(clientID_aux, lidar_handle, vrep.simx_opmode_buffer)
    lidar_data = vrep.simxReadProximitySensor(clientID_aux, lidar_handle, vrep.simx_opmode_buffer)
    #dst = np.linalg.norm(lidar_data)
    print(lidar_data)

