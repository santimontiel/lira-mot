#!/usr/bin/env python

import os
import sys
from PIL import Image as ImagePIL
import numpy as np

# ROS imports

import rospy
import tf # TODO: Upgrade to ROS Noetic 
from geometry_msgs.msg import Transform

if __name__ == '__main__':
    rospy.init_node('t4ac_aux_tf_lidar2radar_ros_node')

    laser_frame = rospy.get_param('/t4ac/frames/laser')
    radar_frame = rospy.get_param('/t4ac/frames/radar')
    
    print("Laser frame: ", laser_frame)
    print("RADAR frame: ", radar_frame)

    listener = tf.TransformListener()

    tf_lidar2radar = Transform()

    pub_tf = rospy.Publisher('t4ac/transform/laser2radar', Transform, queue_size=10)

    while not rospy.is_shutdown():
        try:
            (translation,quaternion) = listener.lookupTransform(laser_frame, radar_frame, rospy.Time(0))

            tf_lidar2radar.translation.x = translation[0]
            tf_lidar2radar.translation.y = translation[1]
            tf_lidar2radar.translation.z = translation[2]

            tf_lidar2radar.rotation.x = quaternion[0]
            tf_lidar2radar.rotation.y = quaternion[1]
            tf_lidar2radar.rotation.z = quaternion[2]
            tf_lidar2radar.rotation.w = quaternion[3]

            pub_tf.publish(tf_lidar2radar)
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue