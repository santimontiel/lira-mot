import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Cluster():
    points: np.ndarray
    center: np.ndarray
    sensor: Optional[str] = 'lidar'
    speed: Optional[float] = 0.0

    def __init__(self, points, sensor):
        if sensor == 'lidar':
            self.points = points
            self.speed = 0.0
        elif sensor == 'radar':
            self.points = points[:, [0,1,2]]
            self.speed = np.mean(points[:,3])
        self.center = np.mean(self.points, axis=0)
        self.sensor = sensor

def euclidean_distance(c1: Cluster, c2: Cluster) -> float:
    return np.linalg.norm(c1.center - c2.center)

def merging_clusters(ll: List[Cluster], d: float, sensor: str) -> List[Cluster]:
    """Function that returns a list of clusters from a list of clusters.
    It merges some clusters if the Euclidean distance between both centers
    is smaller than d"""
    processed = [False for el in ll]                                # Processed cluster flag list
    merged = []                                                     # New merged cluster list
    for i in range(0, len(ll)):                                     # Iterate through all the clusters
        a_counter = 0                                               # Init auxiliary counter
        a_cluster = ll[i]                                           # Init new cluster
        if not processed[i]:                                        # If this cluster has not been processed yet
            for j in range(i+1, len(ll)):                           # Iterate through the rest of the clusters
                if euclidean_distance(a_cluster, ll[j]) <= 1:       # If the distance_centers <= d
                    a_cluster = Cluster(np.concatenate((a_cluster.points, ll[j].points),
                                 axis=0), sensor=sensor)            # Create a new cluster
                    a_counter += 1                                  # Increase aux counter
            merged.append(a_cluster)                                # Append new cluster to merged list
        else:                                                       # If it has been already processed
            continue                                                # Continue
    return merged                                                   # Return new cluster list    
    
@dataclass
class BoundingBox3D:
    points: np.ndarray
    center: np.ndarray
    vertices: np.ndarray
    dimensions: Tuple[float, float, float]
    yaw: float
    sensor: Optional[str] = 'lidar'
    speed: Optional[float] = 0.0

    def __init__(self, points: np.ndarray, speed: float = 0.0):
        self.points = points
        self.center = np.mean(points, axis=0)
        self.vertices = self.get_vertices(self.center)
        self.dimensions = self.get_dimensions(self.vertices)
        self.yaw = self.get_yaw(self.vertices)
        self.speed = speed

    def get_vertices(self, center: np.ndarray) -> np.ndarray:
        """Get the vertices of the bounding box."""
        vertices = np.array(
                [[ 1,  1,  1],
                 [ 1,  1, -1],
                 [ 1, -1,  1],
                 [ 1, -1, -1],
                 [-1,  1,  1],
                 [-1,  1, -1],
                 [-1, -1,  1],
                 [-1, -1, -1]])
        return vertices + center

    def get_dimensions(self, vertices: np.ndarray) -> Tuple[float, float, float]:
        """Get the width, length and height of the bounding box."""
        x_dim = np.max(vertices[0]) - np.min(vertices[0])
        y_dim = np.max(vertices[1]) - np.min(vertices[1])
        z_dim = np.max(vertices[2]) - np.min(vertices[2])
        # w = min(x_dim, y_dim)
        # l = max(x_dim, y_dim)
        l = x_dim
        w = y_dim
        h = z_dim
        return (l, w, h) # x,y,z

    def get_yaw(self, vertices: np.ndarray) -> float:
        """Get the yaw of the bounding box."""
        yaw = np.arctan2(max(vertices[2]) - min(vertices[2]), max(vertices[0]) - min(vertices[0]))
        return yaw

    def format_to_marker_bb_msg(self, center: np.ndarray, dimensions: Tuple[float, float, float], yaw: float) -> List[float]:
        center_as_list = center.tolist()
        return [center_as_list[0], center_as_list[1], center_as_list[2], dimensions[0], dimensions[1], dimensions[2], yaw]
