import sim as vrep
import sys
import numpy as np
import math
 


vrep.simxFinish(-1)
clientID_aux = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
#array=''
x=[]
if clientID_aux!=-1:
    print ("Connected to remote API server")
    err_code,target_handle = vrep.simxGetObjectHandle(clientID_aux,"LaserScannerLaser_2D",vrep.simx_opmode_oneshot_wait)
    i=0
    res,data_u1=vrep.simxGetStringSignal(clientID_aux,'myPtsu1',vrep.simx_opmode_streaming)
    res,data_u2=vrep.simxGetStringSignal(clientID_aux,'myPtsu2',vrep.simx_opmode_streaming)
    print(res)
    laser_beam_u1=np.zeros(90)
    laser_beam_u2=np.zeros(90)
    laser_beam=[]
    while i<100: 
     res,data_u1=vrep.simxGetStringSignal(clientID_aux,'myPtsu1',vrep.simx_opmode_buffer)
     res,data_u2=vrep.simxGetStringSignal(clientID_aux,'myPtsu2',vrep.simx_opmode_buffer)
     data_u1=vrep.simxUnpackFloats(data_u1)
     data_u2=vrep.simxUnpackFloats(data_u2)
     data_u1=np.array(data_u1)
     data_u2=np.array(data_u2)
     
     
     if len(data_u1>1):
        laser_beam_u1=np.zeros(90)
        laser_beam_u2=np.zeros(90)
        for k in range(90):
            laser_beam_u1[k]=data_u1[5+4*k]
            laser_beam_u2[k]=data_u2[5+4*k]    
        laser_beam_u1 = np.insert(laser_beam_u1,90,laser_beam_u2)
        laser_beam_u1[laser_beam_u1<0.01]=2000
        r_min=laser_beam_u1[laser_beam_u1<0.6]    
        b_angles = np.where(laser_beam_u1<0.6) 
        if not len(r_min)==0:
            r=(r_min[0]+r_min[len(r_min)-1])/2
            angles = (b_angles[0][0]+b_angles[0][len(b_angles)-1])/2 
            print(angles)
       
else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")
    
