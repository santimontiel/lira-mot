#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rospy
import math
# import tf

from ros_numpy.point_cloud2 import pointcloud2_to_array
from typing import List
from modules.objects import BoundingBox3D

from std_msgs.msg               import ColorRGBA, Duration
from sensor_msgs.msg            import PointCloud2, PointField
from geometry_msgs.msg          import Point
from visualization_msgs.msg     import Marker, MarkerArray
from jsk_recognition_msgs.msg   import BoundingBox, BoundingBoxArray
from pyquaternion               import Quaternion

from . import geometric_functions

##############################################################################
### CONVERSIONS FROM LiRa-MOT NODE (NUMPY ARRAY) TO ROS MESSAGES #############
##############################################################################

_EPS = np.finfo(float).eps * 4.0

###################################
## From ROS source code (/opt/ros/melodic/lib/python2.7/dist-packages/tf/transformations.py)

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> np.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """

    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

####################################

def lidar2map_coordinates(tf_map2lidar,lidar_bbox):  
  """
  """

  lidar_location = lidar_bbox[:2].tolist()
  lidar_location.append(0) # Z-component (TODO: Improve SORT to 3D-SORT to consider height and z-component)
  lidar_location.append(1) # To homogeneous coordinates
  lidar_location = np.array(lidar_location).reshape(-1,1)
  
  map_location = np.dot(tf_map2lidar,lidar_location).reshape(-1) # == A @ B 

  return map_location

def radar2laser_coordinates(tf_laser2radar,radar_bounding_box):  
  """
  """

  radar_location = list(radar_bounding_box.center)
  radar_location.append(1) # To homogeneous coordinates
  radar_location = np.array(radar_location).reshape(-1,1)
  
  laser_location = np.dot(tf_laser2radar,radar_location).reshape(-1) # == A @ B 

  dim = radar_bounding_box.dimensions
  laser_dim = [dim[0],dim[1],dim[2]]
  laser_location = laser_location[:-1].tolist() + laser_dim + [radar_bounding_box.yaw] + [radar_bounding_box.speed]

  return laser_location

def yaw2quaternion(yaw: float) -> Quaternion:
    """
    """
    return Quaternion(axis=[0,0,1], radians=yaw)

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
        q = yaw2quaternion(bbox.yaw)

        bbox_msg.pose.position.x = bbox.center[0]
        bbox_msg.pose.position.y = bbox.center[1]
        bbox_msg.pose.position.z = bbox.center[2]

        bbox_msg.pose.orientation.x = q[1] 
        bbox_msg.pose.orientation.y = q[2]
        bbox_msg.pose.orientation.z = q[3]
        bbox_msg.pose.orientation.w = q[0]

        bbox_msg.dimensions.x = bbox.dimensions[0] / 10
        bbox_msg.dimensions.y = bbox.dimensions[1] / 10
        bbox_msg.dimensions.z = bbox.dimensions[2] / 10
        bbox_msg.value = 0.0
        bbox_msg.label = 0
        bbox_array.boxes.append(bbox_msg)
    return bbox_array

def marker_bbox_ros_msg(bboxes: List[float], color: str, cb_msg: object, namespace: str) -> MarkerArray:
    """Create a visualization_msgs.MarkerArray from a list of
    T4AC Bounding Boxes (list of 9 param: center, dim, yaw, speed,
    sensor) object.
    """

    colors = {
        "red"       : ColorRGBA(1.0, 0.0, 0.0, 1.0),
        "green"     : ColorRGBA(0.0, 1.0, 0.0, 1.0),
        "blue"      : ColorRGBA(0.0, 0.0, 1.0, 1.0),
        "cyan"      : ColorRGBA(0.0, 1.0, 1.0, 1.0),
        "yellow"    : ColorRGBA(1.0, 1.0, 0.0, 1.0),
        "magenta"   : ColorRGBA(1.0, 0.0, 1.0, 1.0),
        "pink"      : ColorRGBA(1.0, 0.75, 0.79, 1),
        "black"     : ColorRGBA(0.0, 0.0, 0.0, 1.0) 
    }

    bbox_array = MarkerArray()

    for (idx, bbox) in enumerate(bboxes):
        bbox_msg = Marker()
        bbox_msg.header.stamp = cb_msg.header.stamp
        bbox_msg.header.frame_id = 'ego_vehicle/lidar/lidar1'
        bbox_msg.type = 1
    
        bbox_msg.pose.position.x = bbox[0]
        bbox_msg.pose.position.y = bbox[1]
        bbox_msg.pose.position.z = bbox[2]

        q = yaw2quaternion(bbox[6])

        bbox_msg.pose.orientation.x = q[1] 
        bbox_msg.pose.orientation.y = q[2]
        bbox_msg.pose.orientation.z = q[3]
        bbox_msg.pose.orientation.w = q[0]

        bbox_msg.scale.x = bbox[3]
        bbox_msg.scale.y = bbox[4]
        bbox_msg.scale.z = bbox[5]

        bbox_msg.color = colors[color]

        bbox_array.markers.append(bbox_msg)

    return bbox_array

def tracker_to_marker(tracker,color,stamp):
    """
    Fill the obstacle features using real world metrics. Tracker presents a predicted state vector 
    (x,y,l,w,theta, ID)
    """
    tracked_obstacle = Marker()

    tracked_obstacle.header.frame_id = 'ego_vehicle/lidar/lidar1'
    tracked_obstacle.header.stamp = stamp
    tracked_obstacle.ns = "tracked_obstacles"
    tracked_obstacle.action = tracked_obstacle.ADD
    tracked_obstacle.type = 1
        
    tracked_obstacle.id = tracker[5].astype(int)

    tracked_obstacle.pose.position.x = tracker[0] 
    tracked_obstacle.pose.position.y = tracker[1]
    tracked_obstacle.pose.position.z = -0.75 # TODO: Improve this
    
    tracked_obstacle.scale.x = tracker[2]
    tracked_obstacle.scale.y = tracker[3]
    tracked_obstacle.scale.z = 1.5 # TODO: Improve this
    
    quaternion = geometric_functions.quaternion_from_euler(0,0,tracker[4])
    
    tracked_obstacle.pose.orientation.x = quaternion[0]
    tracked_obstacle.pose.orientation.y = quaternion[1]
    tracked_obstacle.pose.orientation.z = quaternion[2]
    tracked_obstacle.pose.orientation.w = quaternion[3]
        
    tracked_obstacle.color.r = color[2]
    tracked_obstacle.color.g = color[1]
    tracked_obstacle.color.b = color[0]
    tracked_obstacle.color.a = 1.0

    tracked_obstacle.lifetime = rospy.Duration(1.0) # 1 second

    return tracked_obstacle
        
def detections_to_marker_array_msg(clusters: list, cb_msg: object, namespace: str, color: str) -> MarkerArray:
    """Create a visualization_msgs.MarkerArray from a np.array list
    of detections.
    """

    colors = {
        "red"       : ColorRGBA(1.0, 0.0, 0.0, 1.0),
        "green"     : ColorRGBA(0.0, 1.0, 0.0, 1.0),
        "blue"      : ColorRGBA(0.0, 0.0, 1.0, 1.0),
        "cyan"      : ColorRGBA(0.0, 1.0, 1.0, 1.0),
        "yellow"    : ColorRGBA(1.0, 1.0, 0.0, 1.0),
        "magenta"   : ColorRGBA(1.0, 0.0, 1.0, 1.0),
        "pink"      : ColorRGBA(1.0, 0.75, 0.79, 1),
        "black"     : ColorRGBA(0.0, 0.0, 0.0, 1.0) 
    }

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
        # marker.color = colors[idx%len(colors)]              # Color
        marker.color = colors[color]              # Color
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