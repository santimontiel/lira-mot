#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

""" LiRa-MOT: Multi-Object Tracking System based on LiDAR and RADAR
    - Author:   Santiago Montiel Marín
    - Date:     September 2021
"""

# -- General purpose imports
import numpy as np
from time import time

# -- ROS imports
import rospy
import ros_numpy
import message_filters
# from tf import TransformListener
from sensor_msgs.msg            import PointCloud2, PointField
from visualization_msgs.msg     import Marker, MarkerArray
from jsk_recognition_msgs.msg   import BoundingBox, BoundingBoxArray
from geometry_msgs.msg          import Point, Transform

# -- Scikit-Learn imports
from sklearn.cluster            import DBSCAN
# from hdbscan                    import HDBSCAN

# -- Open3D imports
import open3d as o3d

# -- Module imports
from modules import (types_helper, detection_helper, geometric_functions, ros_functions, iou_3d_functions)
from modules.objects import (Cluster, BoundingBox3D, euclidean_distance, merging_clusters)

class LiRa():

    #################################################################
    ### INIT FUNCTION ###############################################
    #################################################################
    def __init__(self):
        """ Initialize ROS architecture for LiRa-MOT.
        """
        # Node initialization
        rospy.init_node("lira_mot_node")

        # -- LiDAR and RADAR subscribers
        self.sub_lidar_raw_data         = message_filters.Subscriber("/carla/ego_vehicle/lidar/lidar1/point_cloud", PointCloud2)
        self.sub_radar_raw_data         = message_filters.Subscriber("/carla/ego_vehicle/radar/front/radar_points", PointCloud2)

        # -- Messages synchronization
        self.sync_sensors = message_filters.TimeSynchronizer([self.sub_lidar_raw_data, self.sub_radar_raw_data], queue_size=10)
        self.sync_sensors.registerCallback(self.sensors_cb)

        # -- LiDAR publishers
        self.pub_lidar_filt_cloud       = rospy.Publisher("/t4ac/perception/lidar_filtered_cloud", PointCloud2, queue_size=10)
        self.pub_lidar_plane_cloud      = rospy.Publisher("/t4ac/perception/lidar_plane_cloud", PointCloud2, queue_size=10)
        self.pub_lidar_obstacles_cloud  = rospy.Publisher("/t4ac/perception/lidar_obstacles_cloud", PointCloud2, queue_size=10)
        self.pub_lidar_detections       = rospy.Publisher("/t4ac/perception/lidar_detections", PointCloud2, queue_size=10)
        self.pub_lidar_visualization    = rospy.Publisher("/t4ac/perception/lidar_visualization", MarkerArray, queue_size=10)
        self.pub_lidar_bounding_boxes   = rospy.Publisher("/t4ac/perception/lidar_bounding_boxes", BoundingBoxArray, queue_size=10)
        self.pub_lidar_marker_bb        = rospy.Publisher("/t4ac/perception/lidar_marker_bboxes", MarkerArray, queue_size=10)

        # -- RADAR publishers
        self.pub_radar_detections       = rospy.Publisher("/t4ac/perception/radar_detections", PointCloud2, queue_size=10)
        self.pub_radar_visualization    = rospy.Publisher("/t4ac/perception/radar_visualization", MarkerArray, queue_size=10)
        self.pub_radar_bounding_boxes   = rospy.Publisher("/t4ac/perception/radar_bounding_boxes", MarkerArray, queue_size=10)

        # -- Sensor Fusion publishers

        self.pub_merged_marker_bb       = rospy.Publisher("/t4ac/perception/merged_marker_bboxes", MarkerArray, queue_size=10)
        
        # -- Tf listener to transform radar coordinates to lidar coordinates

        ########### TODO: Improve this directly listening the transform here (Python3 - ROS Noetic)

        aux_tf = rospy.wait_for_message('t4ac/transform/laser2radar', Transform)
        t = aux_tf.translation
        trans = [t.x,t.y,t.z]
        r = aux_tf.rotation
        rot = [r.x,r.y,r.z,r.w]
        rot_matrix = types_helper.quaternion_matrix(rot) # Quaternion to Numpy matrix
            
        self.tf_laser2radar = rot_matrix
        self.tf_laser2radar[:3,3] = self.tf_laser2radar[:3,3] + trans

        # print(">>> TF Laser to RADAR: ", self.tf_laser2radar)
        
        ###########
        
        # self.listener = TransformListener()

        # Multi-Object Tracker

        self.mot_tracker = sort_functions.Sort(max_age=1,min_hits=3)

    #################################################################
    ### SYNC SENSORS CALLBACK #######################################
    #################################################################
    def sensors_cb(self, lidar_data: object, radar_data: object):
        print("----------------------------------------------------------------------------------------")
        print(f">>> LiRa-MOT. Radar and LiDAR processing            >>>")
        print(f"-------------------------------------------------------\n")

        #################################################################
        ### TRANSFORM LISTENER FOR SENSORS ##############################
        #################################################################
        # self.listener.waitForTransform('ego_vehicle/lidar/lidar1', 'ego_vehicle/radar/front', rospy.Time(), rospy.Duration.from_sec(10))

        #################################################################
        ### LIDAR OBJECT DETECTION PIPELINE #############################
        #################################################################
        print(f">>> 1. LiDAR point cloud processing --------------- >>>")
        lt1 = time()

        # 1. Read point cloud msg as numpy array
        lidar_np_orig = types_helper.pointcloud2_to_nparray(lidar_data, "XYZ")

        # 2. Cloud filtering (only looking to frontal 80º)
        lidar_np_filt = detection_helper.lidar_cloud_filtering(lidar_np_orig, 80)
        lidar_ros_filt = types_helper.xyz_array_to_point_cloud_2_msg(lidar_np_filt, stamp=rospy.Time.now(), frame_id="ego_vehicle/lidar/lidar1")
        lidar_o3d_filt = o3d.geometry.PointCloud()
        lidar_o3d_filt.points = o3d.utility.Vector3dVector(lidar_np_filt)

        # 3. Plane segmentation applying RANSAC algorithm
        (lidar_o3d_plane, lidar_o3d_obst) = detection_helper.ransac_plane_segmentation(lidar_o3d_filt, 0.3, 250)
        lidar_ros_obst = detection_helper.o3dpc_to_rospc(lidar_o3d_obst, stamp=rospy.Time.now(), frame_id="ego_vehicle/lidar/lidar1")
        lidar_ros_plane = detection_helper.o3dpc_to_rospc(lidar_o3d_plane, stamp=rospy.Time.now(), frame_id="ego_vehicle/lidar/lidar1")

        # 4. Object detecion using Open3D DBSCAN clustering
        lidar_labels = np.array(lidar_o3d_obst.cluster_dbscan(eps=0.5, min_points=10))
        (lidar_clusters, lidar_indices) = detection_helper.classify_clusters_o3d(lidar_o3d_obst, lidar_labels)

        # 5. Merging lidar clusters if Euclidean distance < 2
        lidar_merged_clusters = merging_clusters(lidar_clusters, 6, sensor='lidar')

        # 6. Building bounding boxes from clusters
        lidar_bb_array = [BoundingBox3D(cluster.points) for cluster in lidar_merged_clusters]

        # 7. Building marker bounding boxes
        lidar_mbb_array = [el.format_to_marker_bb_msg(el.center, el.dimensions, el.yaw) for el in lidar_bb_array]
        lidar_mbb_msg = types_helper.marker_bbox_ros_msg(lidar_mbb_array, "cyan", lidar_data, "lidar_ns")

        # 0. Auxiliary lidar code
        # a. Logging stats to user
        # print(f"Lidar original points: {lidar_np_orig.shape}")
        # print(f"Lidar filtered points: {lidar_np_filt.shape}")
        # print(f"Obstacle points: {lidar_o3d_obst}")
        # if (len(lidar_labels) != 0):
        #     print(f"There are {len(lidar_labels)} elements and {max(lidar_labels)+1} clusters.")
        # else:
        #     print(f"Lidar labels are empty.")
        # print("LiDAR labels: ", lidar_labels)
        print("LiDAR bounding boxes: ", len(lidar_mbb_array))

        # b. Building markers from objects
        lidar_marker_array_msg = types_helper.detections_to_marker_array_msg(lidar_merged_clusters, lidar_data, "lidar_vis")
        lidar_bb_array_msg = types_helper.bounding_boxes_to_ros_msg(lidar_bb_array, lidar_data, "lidar_bb")
        
        # c. publishing messages to ros topics
        self.pub_lidar_filt_cloud.publish(lidar_ros_filt)
        self.pub_lidar_obstacles_cloud.publish(lidar_ros_obst)
        self.pub_lidar_plane_cloud.publish(lidar_ros_plane)
        self.pub_lidar_visualization.publish(lidar_marker_array_msg)
        self.pub_lidar_bounding_boxes.publish(lidar_bb_array_msg)
        self.pub_lidar_marker_bb.publish(lidar_mbb_msg)

        # d. Lidar object detection pipeline ends
        lt2 = time()
        print(f"Time consumed during LiDAR pipeline: {lt2-lt1}\n")


        
        #################################################################
        ### RADAR OBJECT DETECTION PIPELINE #############################
        #################################################################
        print(f">>> 2. Radar point cloud processing --------------- >>>")
        rt1 = time()
            
        # 1. Conversion from sensor_msgs::PointCloud2 to numpy xyzv_array
        self.radar_np_pcd = types_helper.pointcloud2_to_nparray(radar_data, "XYZV")

        # 2. DBSCAN/HDBSCAN algorithm run
        self.dbscan = DBSCAN(eps=0.85, min_samples=18).fit(self.radar_np_pcd)
        # self.dbscan = HDBSCAN(min_samples=18, min_cluster_size=25).fit(self.radarPointCloud)
        rt2 = time()

        # 3. Cluster classification and noise filtering     
        rt3 = time()
        self.indices = list(dict.fromkeys(self.dbscan.labels_))
        if (-1 in self.indices):
            self.indices.remove(-1)
        self.clusters = []
        for i in self.indices:
            self.clusters.append([])
        for (idx, point) in enumerate(self.radar_np_pcd, start=0):
            self.label = self.dbscan.labels_[idx]
            if (self.label != -1):
                self.clusters[self.label].append(point)
        for el in self.clusters:
            el = np.vstack(el)
        radar_clusters = [Cluster(np.asarray(el), sensor='radar') for el in self.clusters]
        rt4 = time()

        # 4. Bounding boxes extraction
        radar_bb_array = [BoundingBox3D(cluster.points, speed=cluster.speed) for cluster in radar_clusters]

        # for el in radar_clusters:
        #     print(el)

        # 0. Auxiliary radar code
        # a. Log info to user
        # print(f"Points: {self.radar_np_pcd.shape}")
        # print(f"DBSCAN took: {rt2-rt1} s")
        # print(f"Number of clusters found: {max(self.dbscan.labels_)+1}")
        # print(f"Number of clustered points: {len(self.dbscan.labels_)-list(self.dbscan.labels_).count(-1)}")
        # print(f"Number of noise points: {list(self.dbscan.labels_).count(-1)}")
        # print(f"Time consumed classifying clusters: {rt4-rt3}")

        # b. Building markers from objects
        detection_markers = types_helper.detections_to_marker_array_msg(radar_clusters, radar_data, "radarspace")
        
        # c. Convert to LiDAR coordinates
        
        radar_lidar_frame_list = []

        for radar_bb in radar_bb_array:
            radar_lidar_frame = types_helper.radar2laser_coordinates(self.tf_laser2radar,radar_bb)
            radar_lidar_frame_list.append(radar_lidar_frame)
        
        radar_bb_array_msg = types_helper.radar_bounding_boxes_to_ros_msg(radar_lidar_frame_list, radar_data, "radar_bb")
        print("RADAR bounding boxes: ", len(radar_bb_array_msg.markers))
        
        radar_mbb_array = [el.format_to_marker_bb_msg(el.center, el.dimensions, el.yaw) for el in radar_bb_array]
        radar_mbb_msg = types_helper.marker_bbox_ros_msg(radar_mbb_array, "magenta", radar_data, "radar_ns")

        # d. publishing messages to ros topics
        self.pub_radar_visualization.publish(detection_markers)
        self.pub_radar_bounding_boxes.publish(radar_mbb_msg)

        # e. Radar object detection pipeline ends
        rtf = time()
        print(f"Time consumed during RADAR pipeline: {rtf-rt1}")

        #################################################################
        ### SENSOR FUSION PIPELINE #############################
        #################################################################

        for i,lidar_obstacle in enumerate(lidar_mbb_array):
            print(f"LiDAR obstacle {i}: {lidar_obstacle}")
            lidar_3d_corners = iou_3d_functions.compute_box_3d(lidar_obstacle)

            for j,radar_obstacle in enumerate(radar_mbb_array):
                print(f"RADAR obstacle {j}: {radar_obstacle}")
                radar_3d_corners = iou_3d_functions.compute_box_3d(radar_obstacle)
                iou3d, _ = iou_3d_functions.box3d_iou(lidar_3d_corners,radar_3d_corners)
                print("iou3d: ", iou3d)


        xyz, lwh, yaw, vel

        # TODO: Transform to Global Coordinates

        # self.pub_merged_marker_bb.publish(merged_obstacles_marker_array)

        #################################################################
        ### MULTI-OBJECT TRACKING PIPELINE #############################
        #################################################################

        merged_objects, types = sort_functions.merged_bboxes_to_xywlthetascore_types(merged_objects)
        print("Merged objects: ", merged_objects)
        print("Types: ", types)
        # trackers = self.mot_tracker.update(METER AQUÍ LOS OBJETOS FUSIONADOS EN FORMATO T4AC BEV DETECTION)

def main() -> None:
    lira_mot_node = LiRa()
    rospy.spin()

if __name__ == "__main__":
    main()