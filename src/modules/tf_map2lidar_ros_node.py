#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import Transform

if __name__ == "__main__":
    rospy.init_node("t4ac_aux_tf_map2lidar_ros_node")

    map_frame = rospy.get_param('/t4ac/frames/map')
    print("Map frame:   ", map_frame)
    laser_frame = rospy.get_param('/t4ac/frames/laser')
    print("Lidar frame: ", laser_frame)

    listener = tf.TransformListener()
    tf_map2lidar = Transform()
    pub_tf_map2lidar = rospy.Publisher("t4ac/transform/map2lidar", Transform, queue_size=10)

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform(map_frame, laser_frame, rospy.Time(0))
            tf_map2lidar.translation.x = trans[0]
            tf_map2lidar.translation.y = trans[1]
            tf_map2lidar.translation.z = trans[2]

            tf_map2lidar.rotation.x = rot[0]
            tf_map2lidar.rotation.y = rot[1]
            tf_map2lidar.rotation.z = rot[2]
            tf_map2lidar.rotation.w = rot[3]

            pub_tf_map2lidar.publish(tf_map2lidar)
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
