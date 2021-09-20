#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from derived_object_msgs.msg import (Object, ObjectArray)
from tf2_msgs.msg import TFMessage
import csv

class TrackingEvaluator:
    
    def __init__(self):

        # Control flags
        self.serialize = False

        # Initializing ROS Node
        rospy.init_node('eval_tracking_groundtruth_node')

        # Subscribing to CARLA Objects
        self.sub_carla_objects = rospy.Subscriber(
            "/carla/objects", ObjectArray, self.cb_carla_objects
        )
        self.sub_tf_frames = rospy.Subscriber(
            "/tf", TFMessage, self.cb_tf_lidar_2_map
        )

        # File to save the results
        if self.serialize:
            self.file = open("/home/santi/Workspace/ros_ws/src/lira-mot/eval_tracking.csv", "w")
            self.writer = csv.writer(self.file)

    def cb_tf_lidar_2_map(self, msg):
        pass
        '''
        for transform in msg.transforms:
            if transform.child_frame_id == "ego_vehicle/lidar/lidar1":
                # print(transform)
                t = transform.transform.translation
                translation = (t.x, t.y, t.z)
                r = transform.transform.rotation
                rotation = (r.x, r.y, r.z, r.w)
                print('\033[94m' + '\033[1m' + "New transform:" + '\033[0m')
                print(f"Translation is {translation} and rotation is {rotation}")
                '''

    def cb_carla_objects(self, msg):
        '''
        print(msg.objects)
        '''
        for object in msg.objects:
            print(f"{object.id} is {object.classification}")
        
            if (object.classification == 4):                # If it is a pedestrian
                print('\033[94m' + '\033[1m' + "ID is: " + str(object.id) + '\033[0m')                            # Print the ID of the pedestrian
                print(object.pose.position.x)                  # Print the x coordinate of the pedestrian
                print(object.pose.position.y)                  # Print the y coordinate of the pedestrian
                print(object.pose.position.z)                  # Print the z coordinate of the pedestrian
                if (object.id == 261):
                    x = object.pose.position.x
                    y = object.pose.position.y
                    z = object.pose.position.z
                
        # Serialize results
        if self.serialize: self.writer.writerow([
            x,                          # X-coordinate of the pedestrian
            y,                          # Y-coordinate of the pedestrian
            z                           # Z-coordinate of the pedestrian
            ])
        


def main(args=None):
    evaluator = TrackingEvaluator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main()