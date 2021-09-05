#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple
import open3d as o3d
import numpy as np
import copy
import rospy, ros_numpy
from sensor_msgs.msg    import PointCloud2, PointField

# -- Scikit-Learn imports
from sklearn.cluster    import DBSCAN

def rospc_to_o3dpc(rospc, remove_nans=False):
    """ covert ros point cloud to open3d point cloud
    Args: 
        rospc (sensor.msg.PointCloud2): ros point cloud message
        remove_nans (bool): if true, ignore the NaN points
    Returns: 
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
    """
    field_names = [field.name for field in rospc.fields]
    is_rgb = 'rgb' in field_names
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(rospc)
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    if is_rgb:
        cloud_npy = np.zeros(cloud_array.shape + (4,), dtype=np.float)
    else: 
        cloud_npy = np.zeros(cloud_array.shape + (3,), dtype=np.float)
    
    cloud_npy[...,0] = cloud_array['x']
    cloud_npy[...,1] = cloud_array['y']
    cloud_npy[...,2] = cloud_array['z']
    o3dpc = open3d.geometry.PointCloud()

    if len(np.shape(cloud_npy)) == 3:
        cloud_npy = np.reshape(cloud_npy[:, :, :3], [-1, 3], 'F')
    o3dpc.points = open3d.utility.Vector3dVector(cloud_npy[:, :3])

    if is_rgb:
        rgb_npy = cloud_array['rgb']
        rgb_npy.dtype = np.uint32
        r = np.asarray((rgb_npy >> 16) & 255, dtype=np.uint8)
        g = np.asarray((rgb_npy >> 8) & 255, dtype=np.uint8)
        b = np.asarray(rgb_npy & 255, dtype=np.uint8)
        rgb_npy = np.asarray([r, g, b])
        rgb_npy = rgb_npy.astype(np.float)/255
        rgb_npy = np.swapaxes(rgb_npy, 0, 1)
        o3dpc.colors = open3d.utility.Vector3dVector(rgb_npy)
    return o3dpc

def o3dpc_to_rospc(o3dpc, frame_id=None, stamp=None):
    """ convert open3d point cloud to ros point cloud
    Args:
        o3dpc (open3d.geometry.PointCloud): open3d point cloud
        frame_id (string): frame id of ros point cloud header
        stamp (rospy.Time): time stamp of ros point cloud header
    Returns:
        rospc (sensor.msg.PointCloud2): ros point cloud message
    """

    cloud_npy = np.asarray(copy.deepcopy(o3dpc.points))
    is_color = o3dpc.colors
        

    n_points = len(cloud_npy[:, 0])
    if is_color:
        data = np.zeros(n_points, dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('rgb', np.uint32)
        ])
    else:
        data = np.zeros(n_points, dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
            ])
    data['x'] = cloud_npy[:, 0]
    data['y'] = cloud_npy[:, 1]
    data['z'] = cloud_npy[:, 2]
    
    if is_color:
        rgb_npy = np.asarray(copy.deepcopy(o3dpc.colors))
        rgb_npy = np.floor(rgb_npy*255) # nx3 matrix
        rgb_npy = rgb_npy[:, 0] * BIT_MOVE_16 + rgb_npy[:, 1] * BIT_MOVE_8 + rgb_npy[:, 2]  
        rgb_npy = rgb_npy.astype(np.uint32)
        data['rgb'] = rgb_npy

    rospc = ros_numpy.msgify(PointCloud2, data)
    if frame_id is not None:
        rospc.header.frame_id = frame_id

    if stamp is None:
        rospc.header.stamp = rospy.Time.now()
    else:
        rospc.header.stamp = stamp
    rospc.height = 1
    rospc.width = n_points
    rospc.fields = []
    rospc.fields.append(PointField(
                            name="x",
                            offset=0,
                            datatype=PointField.FLOAT32, count=1))
    rospc.fields.append(PointField(
                            name="y",
                            offset=4,
                            datatype=PointField.FLOAT32, count=1))
    rospc.fields.append(PointField(
                            name="z",
                            offset=8,
                            datatype=PointField.FLOAT32, count=1))    

    if is_color:
        rospc.fields.append(PointField(
                        name="rgb",
                        offset=12,
                        datatype=PointField.UINT32, count=1))    
        rospc.point_step = 16
    else:
        rospc.point_step = 12
    
    rospc.is_bigendian = False
    rospc.row_step = rospc.point_step * n_points
    rospc.is_dense = True
    return rospc


def lidar_cloud_filtering(cloud: np.ndarray, fov: float) -> np.ndarray:
    """
    Function that filters a point cloud depending of a desired field
    of view. Resulting point cloud will have horizontal FOV of 
    +-fov/2.
    """
    min_fov = -(fov/2)*(3.14/180)
    max_fov = (fov/2)*(3.14/180)
    mask = ((cloud[:,2] > -2.00) 
        & (np.arctan2(cloud[:,1], cloud[:,0]) > min_fov) 
        & (np.arctan2(cloud[:,1], cloud[:,0]) < max_fov))
    cloud = cloud[mask]
    return cloud

def voxel_grid_downsample(cloud: object, voxel_size: float) -> object:
    return cloud.voxel_down_sample(voxel_size=voxel_size)

def ransac_plane_segmentation(cloud: object, thresh: int, n_iter: int) -> Tuple[object, object]:
    """
    Performs RANSAC plane segmentation over a point cloud and returns
    a tuple with two clouds: one composed by the inliers (plane) and
    other one composed by the outliers (obstacles). Based on Open3d
    library functions. Plane model not used.
    """
    plane_model, inliers = cloud.segment_plane(
        distance_threshold=thresh,
        ransac_n=3,
        num_iterations=n_iter)
    plane_cloud = cloud.select_by_index(inliers)
    obstacles_cloud = cloud.select_by_index(inliers, invert=True)
    return (plane_cloud, obstacles_cloud)


def clustering_dbscan_o3d():
    """
    Performs clustering over obstacles in a point cloud applying
    Open3D implementation of DBSCAN algorithm.
    """
    pass

def clustering_knn_o3d():
    pass

def clustering_dbscan_sk(cloud: object, eps: int, min_samples: int) -> object:
    """
    Performs clustering over obstacles in a point cloud applying
    Scikit-Learn implementation of DBSCAN algorithm.
    """
    return DBSCAN(eps=eps, min_samples=min_samples).fit(cloud)

def get_dbscan_stats_sk(dbscan: object, mode: str) -> Tuple[int, int, int]:
    """
    Function that obtains interesting statistics about the clustering
    performed over a point cloud using Scikit-Learn implementation of
    DBSCAN algorithm. 
    Returns:
        - n_clusters:       Number of clusters
        - n_points:         Number of total points
        - n_valid_points:   Number of points that belongs to a cluster
        - n_noise_points:   Number of points that does not belong to 
                                any cluster
    """
    n_clusters      = max(dbscan.labels_) + 1
    n_points        = len(dbscan.labels_)
    n_noise_points  = list(dbscan.labels_).count(-1)
    n_valid_points  = n_points - n_noise_points
    if mode == "var":
        return (n_clusters, n_points, n_valid_points, n_noise_points)
    elif mode == "console":
        return print(f"""\n\t.:>>> POINT CLOUD STATS AFTER DBSCAN <<<:.\n
        - Number of cluster:            {n_clusters}
        - Number of total points:       {n_points}
        - Number of clustered points:   {n_valid_points}
        - Number of noise points:       {n_noise_points}
        """)
    else:
        return print(f"Not valid mode.")

from modules.objects import Cluster

def classify_clusters_sk(cloud: object, labels: np.ndarray) -> list: 
    """
    Function that classify the labels obtained after Scikit-Learn
    clustering with DBSCAN algorithm and returns a list of clusters
    and other stats.
    """
    """
    indices = list(dict.fromkeys(labels))
    if (-1 in indices):
        indices.remove(-1)
    clusters = [[] for i in indices]
    for (i, point) in enumerate(cloud, start=0):
        if (labels[i] != -1):
            clusters[labels[i]].append(point)  
    return (clusters, indices)
    """
    indices = list(dict.fromkeys(labels))
    if (-1 in indices):
        indices.remove(-1)
    clusters = []
    for i in indices:
        clusters.append([])
    for (idx, point) in enumerate(cloud, start=0):
        label = labels[idx]
        if (label != -1):
            clusters[label].append(point)
    for el in clusters:
        el = np.vstack(el)
    radar_clusters = [Cluster(np.asarray(el), sensor='radar') for el in clusters]
    return (radar_clusters, indices)

def classify_clusters_o3d(cloud: object, labels: np.ndarray) -> Tuple[list, list]:
    """
    Function that classify the labels obtained after Open3D
    clustering with DBSCAN algorithm and returns a list of clusters
    and other stats.
    """
    cloud_np = np.asarray(cloud.points)
    indices = list(dict.fromkeys(labels))
    if (-1 in indices):
        indices.remove(-1)
    clusters = [[] for i in indices]
    for (i, point) in enumerate(cloud_np, start=0):
        if (labels[i] != -1):
            clusters[labels[i]].append(point)
    for el in clusters:
        el = np.vstack(el)
    cluster_objects = [Cluster(np.asarray(el), sensor='lidar') for el in clusters]

    return (cluster_objects, indices)
    
