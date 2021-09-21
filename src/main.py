#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

""" LiRa-MOT: Multi-Object Tracking System based on LiDAR and RADAR
    - Author:   Santiago Montiel Marín
    - Date:     September 2021
"""

# -- General purpose imports
import numpy as np
from time import time
import csv
import math

# -- ROS imports
import rospy
import ros_numpy
import message_filters
# from tf import TransformListener
from sensor_msgs.msg            import PointCloud2, PointField
from visualization_msgs.msg     import Marker, MarkerArray
from jsk_recognition_msgs.msg   import BoundingBox, BoundingBoxArray
from geometry_msgs.msg          import Point, Transform
from derived_object_msgs.msg    import Object, ObjectArray
from carla_msgs.msg             import CarlaEgoVehicleInfo
from nav_msgs.msg               import Odometry

# -- Scikit-Learn imports
from sklearn.cluster            import DBSCAN
# from hdbscan                    import HDBSCAN

# -- Open3D imports
import open3d as o3d

# -- Module imports
from modules import (types_helper, detection_helper, geometric_functions, ros_functions, iou_3d_functions, sort_functions)
from modules.objects import (Cluster, BoundingBox3D, euclidean_distance, merging_clusters)

class LiRaMOT():

    #################################################################
    ### INIT FUNCTION ###############################################
    #################################################################
    def __init__(self):
        """ Initialize ROS architecture for LiRa-MOT.
        """
        ego_vehicle_info = rospy.wait_for_message('/carla/ego_vehicle/vehicle_info', CarlaEgoVehicleInfo)
        self.ego_vehicle_id = ego_vehicle_info.id
        self.sub_carla_objects          = rospy.Subscriber("/carla/objects", ObjectArray, self.carla_objects_callback)
        self.ego_vehicle_velocity = 0
        self.sub_ego_vehicle_pose = rospy.Subscriber("/t4ac/localization/pose", Odometry, self.ego_vehicle_pose_callback)

        # -- Debug
        self.debug_lidar = False
        self.debug_radar = False
        self.debug_fusion = False
        self.debug_mot = False
        self.serialize = True
        self.carla_objects_flag = False
        if self.serialize:
            self.file = open("/home/robesafe/t4ac_ws/src/t4ac_architecture/lira-mot/results/eval_tracking_2.csv", "w") # File to save the results
            self.writer = csv.writer(self.file)

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

        # -- Tf listener to transform lidar coordinates to map coordinates
        self.sub_tfmap2lidar = rospy.Subscriber("/t4ac/transform/map2lidar", Transform, self.map2lidar_callback)

        # Multi-Object Tracker
        self.mot_tracker = sort_functions.Sort(max_age=5,min_hits=1, iou_threshold=0.001)
        self.colours = np.random.rand(32,3)

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
        self.pub_merged_marker_bb       = rospy.Publisher("/t4ac/perception/merged_bounding_bboxes", MarkerArray, queue_size=10)

        # -- Multi-Object Tracking publishers
        self.pub_mot_marker_bb          = rospy.Publisher("/t4ac/perception/mot_bounding_bboxes", MarkerArray, queue_size=10)

        # -- File serializer CSV
        # self.file = open("/home/robesafe/santi_data.csv", "w")
        # self.writer = csv.writer(self.file)
        

        ###########
        
        # self.listener = TransformListener() 

    def ego_vehicle_pose_callback(self, msg):
        """
        """

        self.ego_vehicle_velocity = math.sqrt(pow(msg.twist.twist.linear.x,2)+pow(msg.twist.twist.linear.y,2))
    def carla_objects_callback(self, msg):
        """
        """
        for carla_object in msg.objects:
            # print(f"{carla_object.id} is {carla_object.classification}")
            # print('\033[94m' + '\033[1m' + "ID is: " + str(carla_object.id) + '\033[0m')                            # Print the ID of the pedestrian
            # print(carla_object.pose.position.x)                  # Print the x coordinate of the pedestrian
            # print(carla_object.pose.position.y)                  # Print the y coordinate of the pedestrian
            # print(carla_object.pose.position.z)                  # Print the z coordinate of the pedestrian
            if (carla_object.id != self.ego_vehicle_id):
                self.gt_frame = msg.header.seq
                self.gtx = carla_object.pose.position.x
                self.gty = carla_object.pose.position.y
                self.gtz = carla_object.pose.position.z
                self.vel_lin = math.sqrt(pow(carla_object.twist.linear.x,2)+pow(carla_object.twist.linear.y,2))
                self.carla_objects_flag = True
    def map2lidar_callback(self, msg):
        """
        """

        t = msg.translation
        trans = [t.x,t.y,t.z]
        r = msg.rotation
        rot = [r.x,r.y,r.z,r.w]
        rot_matrix = types_helper.quaternion_matrix(rot) # Quaternion to Numpy matrix
            
        self.tf_map2lidar = rot_matrix
        self.tf_map2lidar[:3,3] = self.tf_map2lidar[:3,3] + trans

    #################################################################
    ### SYNC SENSORS CALLBACK #######################################
    #################################################################
    def sensors_cb(self, lidar_data: object, radar_data: object):
        print("----------------------------------------------------------------------------------------")
        if self.carla_objects_flag: print("Object gt: ", self.gtx, self.gty, self.gtz, self.vel_lin)
        # print(f">>> LiRa-MOT. Radar and LiDAR processing            >>>")
        # print(f"-------------------------------------------------------\n")

        #################################################################
        ### TRANSFORM LISTENER FOR SENSORS ##############################
        #################################################################
        # self.listener.waitForTransform('ego_vehicle/lidar/lidar1', 'ego_vehicle/radar/front', rospy.Time(), rospy.Duration.from_sec(10))

        #################################################################
        ### LIDAR OBJECT DETECTION PIPELINE #############################
        #################################################################
        if self.debug_lidar: print(f">>> 1. LiDAR point cloud processing --------------- >>>")
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
        lidar_mbb_array = [el.format_to_marker_bb_msg() for el in lidar_bb_array]
        lidar_mbb_msg = types_helper.marker_bbox_ros_msg(lidar_mbb_array, "cyan", lidar_data, "lidar_ns")

        # 0. Auxiliary lidar code
        # a. Logging stats to user
        if self.debug_lidar:
            print(f"Lidar original points: {lidar_np_orig.shape}")
            print(f"Lidar filtered points: {lidar_np_filt.shape}")
            print(f"Obstacle points: {lidar_o3d_obst}")
            if (len(lidar_labels) != 0):
                print(f"There are {len(lidar_labels)} elements and {max(lidar_labels)+1} clusters.")
            else:
                print(f"Lidar labels are empty.")
            print("LiDAR labels: ", lidar_labels)
            print("LiDAR Bounding Boxes: ", len(lidar_mbb_msg.markers))

        # b. Building markers from objects
        lidar_marker_array_msg = types_helper.detections_to_marker_array_msg(lidar_merged_clusters, lidar_data, "lidar_vis", "blue")
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
        if self.debug_lidar: print(f"Time consumed during LiDAR pipeline: {lt2-lt1}\n")


        
        #################################################################
        ### RADAR OBJECT DETECTION PIPELINE #############################
        #################################################################
        if self.debug_radar: print(f">>> 2. Radar point cloud processing --------------- >>>")
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
        if self.debug_radar:
            print(f"Points: {self.radar_np_pcd.shape}")
            print(f"DBSCAN took: {rt2-rt1} s")
            print(f"Number of clusters found: {max(self.dbscan.labels_)+1}")
            print(f"Number of clustered points: {len(self.dbscan.labels_)-list(self.dbscan.labels_).count(-1)}")
            print(f"Number of noise points: {list(self.dbscan.labels_).count(-1)}")
            print(f"Time consumed classifying clusters: {rt4-rt3}")

        # b. Building markers from objects
        detection_markers = types_helper.detections_to_marker_array_msg(radar_clusters, radar_data, "radarspace", "pink")
        
        # c. Convert to LiDAR coordinates

        radar_lidar_frame_list = []

        for radar_bb in radar_bb_array:
            radar_lidar_frame = types_helper.radar2laser_coordinates(self.tf_laser2radar,radar_bb)
            radar_lidar_frame_list.append(radar_lidar_frame)
    
        if self.debug_radar: print("RADAR bounding boxes: ", len(radar_lidar_frame_list))
        
        # radar_mbb_array = [el.format_to_marker_bb_msg(el.center, el.dimensions, el.yaw) for el in radar_bb_array]
        # radar_mbb_array = [el.format_to_marker_bb_msg() for el in radar_bb_array]
        
        radar_mbb_msg = types_helper.marker_bbox_ros_msg(radar_lidar_frame_list, "magenta", radar_data, "radar_ns")
        # radar_mbb_msg = types_helper.marker_bbox_ros_msg(radar_mbb_array, "magenta", radar_data, "radar_ns")

        # d. publishing messages to ros topics
        self.pub_radar_visualization.publish(detection_markers)
        self.pub_radar_bounding_boxes.publish(radar_mbb_msg)

        # e. Radar object detection pipeline ends
        rtf = time()
        if self.debug_radar: print(f"Time consumed during RADAR pipeline: {rtf-rt1}\n")

        #################################################################
        ### SENSOR FUSION PIPELINE #############################
        #################################################################
        if self.debug_fusion: print(f">>> 3. Sensor Fusion processing --------------- >>>")
        sft1 = time()

        merged_bboxes = []
        
        for i,lidar_obstacle in enumerate(lidar_mbb_array):
            if self.debug_fusion:
                print(".................")
                print(f"LiDAR obstacle {i}: {lidar_obstacle}")
            lidar_3d_corners = iou_3d_functions.compute_box_3d(lidar_obstacle)
            iou3d_min = 0.01

            for j,radar_obstacle in enumerate(radar_lidar_frame_list):
                radar_3d_corners = iou_3d_functions.compute_box_3d(radar_obstacle)
                iou3d, _ = iou_3d_functions.box3d_iou(lidar_3d_corners,radar_3d_corners)
                if self.debug_fusion: 
                    print(f"RADAR obstacle {j}: {radar_obstacle}")
                    print("iou3d: ", iou3d)
                merged_bbox = []
                if iou3d > iou3d_min:
                    iou3d_min = iou3d
                    merged_bbox = [lidar_obstacle[0],lidar_obstacle[1],lidar_obstacle[2], # x,y,z
                                   lidar_obstacle[3],lidar_obstacle[4],lidar_obstacle[5], # l,w,h
                                   lidar_obstacle[6],radar_obstacle[7], "generic_object"] # TODO: Improve object type and velocity
                if merged_bbox: # Not empty
                    print("\033[1;31m"+"Merged bbox: "+'\033[0;m', merged_bbox)
                    merged_bboxes.append(merged_bbox)

        merged_bboxes_msg = types_helper.marker_bbox_ros_msg(merged_bboxes, "black", lidar_data, "sensor_fusion_ns")
        self.pub_merged_marker_bb.publish(merged_bboxes_msg)

        sft2 = time()
        if self.debug_fusion: print(f"Time consumed during Sensor Fusion pipeline: {sft2-sft1}\n")

        #################################################################
        ### MULTI-OBJECT TRACKING PIPELINE #############################
        #################################################################
        if self.debug_mot: print(f">>> 4. Multi-Object Tracking processing --------------- >>>")
        mott1 = time()

        merged_objects, types = sort_functions.merged_bboxes_to_xylwthetascore_types(merged_bboxes)
        if self.debug_mot: print("Merged objects: ", merged_objects)

        trackers, types, vels = self.mot_tracker.update(merged_objects, types, self.debug_mot)

        if self.debug_mot: print("\033[1;35m"+"Final Trackers: "+'\033[0;m', trackers)
        stamp = lidar_data.header.stamp
        tracker_marker_list = MarkerArray()

        for tracker in trackers:
            color = self.colours[tracker[5].astype(int)%32]
            tracker_marker = types_helper.tracker_to_marker(tracker,color,stamp)
            tracker_marker_list.markers.append(tracker_marker)

        map_based_trackers = [types_helper.lidar2map_coordinates(self.tf_map2lidar,tracker) for tracker in trackers]
        print("Trackers: ", map_based_trackers)
        print("Types: ", types)
        print("Velocities: ", vels)
        if self.debug_mot: print("MOT markers: ", len(tracker_marker_list.markers))
        self.pub_mot_marker_bb.publish(tracker_marker_list)

        mott2 = time()
        if self.debug_mot: print(f"Time consumed during Multi-Object Tracking pipeline: {mott2-mott1}")
                        
        # Serialize results

        if len(map_based_trackers) > 0 and sum(merged_objects[0]) > 0 and len(vels) > 0:
            tr_x = map_based_trackers[0][0]
            tr_y = map_based_trackers[0][1]
            tr_vel = self.ego_vehicle_velocity + vels[0] # UFFFFFF MIS OJOSSSSSSS
        else:
            tr_x = 2000
            tr_y = 2000
            tr_vel = 2000

        if self.serialize: 
            self.writer.writerow([
            self.gt_frame,                     # Frame (Seq)
            self.gtx,                          # X-coordinate of the pedestrian
            self.gty,                          # Y-coordinate of the pedestrian
            self.vel_lin,                      # Linear velocity (Module xy) 
            tr_x,
            tr_y,
            tr_vel      
            ])

        # TODO: 

        # 1. yaw siempre 0 del lidar?
        # 2. Corregir coche girado inicialmente (casi siempre), es el segundo que aparece en campus_demo_final_v2.bag
        # 3. Almacenar en .csv el siguiente formato
        # 4. Integrar la velocidad del radar en el filtro bayesiano (descomponiendo), no solo propagando
        # 5. Hacer BIEN la compensación de velocidades
        # Groundtruth: Frame_idIDxyvel (xy en globales)
        # Trackers:    Frame_idIDxyvel
        #################################################################
        ### TIME ANALYSIS ###############################################
        #################################################################
        '''
        self.writer.writerow([
            mott2-lt1,                                              # Total callback time
            lt2-lt1,                                                # LiDAR Object Detection pipeline time
            rt2-rt1,                                                # RADAR Object Detection pipeline time
            sft2-sft1,                                              # Sensor Fusion pipeline time
            mott2-mott1                                             # Multi-Object Tracking pipeline time
        ])
        '''


def main() -> None:
    # Node initialization
    rospy.init_node("lira_mot_node")
    lira_mot_node = LiRaMOT()
    rospy.spin()

if __name__ == "__main__":
    main()