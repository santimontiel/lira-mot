#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import tf

from ros_numpy.point_cloud2 import pointcloud2_to_array
from typing import List
from modules.objects import BoundingBox3D

from std_msgs.msg               import ColorRGBA, Duration
from sensor_msgs.msg            import PointCloud2, PointField
from geometry_msgs.msg          import Point
from visualization_msgs.msg     import Marker, MarkerArray
from jsk_recognition_msgs.msg   import BoundingBox, BoundingBoxArray

##############################################################################
### CONVERSIONS FROM LiRa-MOT NODE (NUMPY ARRAY) TO ROS MESSAGES #############
##############################################################################

colors = [
    ColorRGBA(1.0, 0.0, 0.0, 1.0),          # Red
    ColorRGBA(0.0, 1.0, 0.0, 1.0),          # Green
    ColorRGBA(0.0, 0.0, 1.0, 1.0),          # Blue
    ColorRGBA(0.0, 1.0, 1.0, 1.0),          # Cyan
    ColorRGBA(1.0, 1.0, 0.0, 1.0),          # Yellow
    ColorRGBA(1.0, 0.0, 1.0, 1.0),          # Magenta
    ColorRGBA(0.0, 0.0, 0.0, 1.0)           # Black
]

def bounding_boxes_to_ros_msg(bboxes: List[BoundingBox3D], cb_msg: object, namespace: str) -> BoundingBoxArray:
    """Create a jsk_recognition_msgs.BoundingBoxArray from a list of
    BoundingBox3D dataclass object.
    """
    bbox_array = BoundingBoxArray()
    bbox_array.header.stamp = rospy.Time.now()
    bbox_array.header.frame_id = 'ego_vehicle/lidar/lidar1'
    for (idx, bbox) in enumerate(bboxes):
        bbox_msg = BoundingBox()
        bbox_msg.header.stamp = cb_msg.header.stamp
        bbox_msg.header.frame_id = 'ego_vehicle/lidar/lidar1'
        quat = tf.transformations.quaternion_from_euler(ai=0, aj=0, ak=0)
        bbox_msg.pose.position.x = bbox.center[0]
        bbox_msg.pose.position.y = bbox.center[1]
        bbox_msg.pose.position.z = bbox.center[2]
        bbox_msg.pose.orientation.w = quat[0]
        bbox_msg.pose.orientation.x = quat[1]
        bbox_msg.pose.orientation.y = quat[2]
        bbox_msg.pose.orientation.z = bbox.yaw
        bbox_msg.dimensions.x = bbox.dimensions[0] / 5
        bbox_msg.dimensions.y = bbox.dimensions[1] / 5
        bbox_msg.dimensions.z = bbox.dimensions[2] / 5
        bbox_msg.value = 0.0
        bbox_msg.label = 0
        bbox_array.boxes.append(bbox_msg)
    return bbox_array
        

def detections_to_marker_array_msg(clusters: list, cb_msg: object, namespace: str) -> MarkerArray:
    """Create a visualization_msgs.MarkerArray from a np.array list
    of detections.
    """
    marker_array = MarkerArray()
    for (idx, cloud) in enumerate(clusters):
        marker = Marker()                                   # Creation of a single Marker
        marker.header.stamp = cb_msg.header.stamp           # Header
        marker.header.frame_id = cb_msg.header.frame_id
        marker.ns = namespace                               # Namespace
        marker.id = 0                                       # Id
        marker.type = 8                                     # Type
        marker.action = 0                                   # Action
        marker.scale.x = 0.5                                # Scale
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color = colors[idx%len(colors)]              # Color
        marker.points = [Point(point[0], point[1], point[2]) for point in cloud.points] # Points
        marker_array.markers.append(marker)                 # Add Marker to MarkerArray

    return marker_array

def xyz_array_to_point_cloud_2_msg(points_sum, stamp=None, frame_id=None):
    """
    Create a sensor_msgs.PointCloud2 from an array of points.
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

##############################################################################
### CONVERSIONS FROM CARLA ROS BRIDGE TO NUMPY ARRAYS ########################
##############################################################################

def get_points(cloud_array: object, cloud_type: str, remove_nans=True, dtype=np.float):
    '''
    Pulls out x, y, and z columns from the cloud recordarray, and returns
	a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    if cloud_type == "XYZ":
        points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    elif cloud_type == "XYZV":
        points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    if cloud_type == "XYZ":
        pass
    elif cloud_type == "XYZV":
        points[...,3] = cloud_array['Velocity']

    return points

def pointcloud2_to_nparray(cloud_msg: object, cloud_type: str, remove_nans=True) -> np.ndarray:
    """
    Suitable for Lidar and Radar sensor detections coming from CARLA simulator.
    Create a XYZ (nx3) or XYZV (nx4) np.ndarray from a sensor_msgs.PointCloud2.
    Modified version of ros_numpy.point_cloud2.pointcloud2_to_xyz_array
    """
    return get_points(pointcloud2_to_array(cloud_msg), cloud_type, remove_nans=remove_nans)