#!/usr/bin/env python
# ROS python API
from gazebo_connections import GazeboConnection
from fcu_modes import fcuModes
import rospy
from std_srvs.srv    import Empty, EmptyRequest  
from sensor_msgs.msg import Imu
# 3D point & Stamped Pose msgs
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from std_srvs.srv import *
import time


class Controller:
    # initialization method
    def __init__(self):

        self.gazebo = GazeboConnection()
        self.modes = fcuModes()
        self.local_velocity = TwistStamped()
        self.rate = rospy.Rate(20.0)
        # Drone state
        self.state = State()
        self.imu_data = Imu()
        # Instantiate a setpoints message
        self.sp = PositionTarget()
        # set the flag to use position setpoints and yaw angle
        self.sp.type_mask = int('010111111000', 2)
        # LOCAL_NED
        self.sp.coordinate_frame = 1
        
        # We will fly at a fixed altitude for now
        # Altitude setpoint, [meters]
        self.ALT_SP = 1.0
        # update the setpoint message with the required altitude
        self.sp.position.z = self.ALT_SP
        # Step size for position update
        self.STEP_SIZE = 2.0
		# Fence. We will assume a square fence for now
        self.FENCE_LIMIT = 5.0

        # A Message for the current local position of the drone
        self.local_pos = Point(0.0, 0.0, 1.0)
        # speed of the drone is set using MPC_XY_CRUISE parameter in MAVLink
        # using QGroundControl. By default it is 5 m/s.

	# Callbacks

    ## local position callback
    def posCb(self, msg):
        self.local_pos.x = msg.pose.position.x
        self.local_pos.y = msg.pose.position.y
        self.local_pos.z = msg.pose.position.z
    
    def imu_callback(self,msg):
        self.imu_data.orientation.x = msg.orientation.x
        self.imu_data.orientation.y = msg.orientation.y
        self.imu_data.orientation.z = msg.orientation.z

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg


def main():

    # initiate node
    rospy.init_node('new_reset_try', anonymous=True)
    # controller object
    cnt = Controller()

    # ROS loop rate
    cnt.rate = rospy.Rate(20.0)

    # Subscribe to drone state
    rospy.Subscriber('mavros/state', State, cnt.stateCb)

    rospy.Subscriber('mavros/local_position/pose', PoseStamped, cnt.posCb)
    rospy.Subscriber('mavros/imu/data', Imu, cnt.imu_callback)

    # Setpoint publisher    
    sp_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size=5)
    vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=1)
    


    # Make sure the drone is armed
    while not cnt.state.armed:
        cnt.modes.setArm()
        cnt.rate.sleep()

    # set in takeoff mode and takeoff to default altitude (3 m)
    #cnt.modes.setTakeoff()
    #cnt.rate.sleep()

    # We need to send few setpoint messages, then activate OFFBOARD mode, to take effect
    k=0
    while cnt.local_pos.z<0.9:
        cnt.sp.position.z = 1
        cnt.modes.setOffboardMode()
        sp_pub.publish(cnt.sp)
        cnt.rate.sleep()
        k = k + 1

    # activate OFFBOARD mode

    time.sleep(1)
    # ROS main loop
    while not rospy.is_shutdown():
        cnt.local_velocity.twist.linear.x = 1
        cnt.local_velocity.twist.linear.y = 0
        print(cnt.imu_data.orientation.x,cnt.imu_data.orientation.y,cnt.imu_data.orientation.z)
        vel_pub.publish(cnt.local_velocity)
        cnt.rate.sleep()


if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
