#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 02:15:37 2021

@author: Carlos Gomez-Huelamo

Code to 

Communications are based on ROS (Robot Operating Sytem)

Inputs: 

Outputs:  

Note that 

"""

import numpy as np
import math

import rospy
import geometry_msgs.msg
import visualization_msgs.msg

import tracking_functions

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.001):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = tracking_functions.associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = tracking_functions.KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

# TODO: Pasar de BEV detection a x1y1x2y2score

def bbox_to_xywh_cls_conf(self,detections_rosmsg,img):
    """
    """

    bboxes = []
    types = []
    k = 0

    # Evaluate detections

    for bbox_object in detections_rosmsg.bev_detections_list:
        if (bbox_object.score >= self.detection_threshold):

            bbox_object.x_corners = np.array(bbox_object.x_corners) # Tuple to np.ndarray
            bbox_object.y_corners = np.array(bbox_object.y_corners) 

            # Gaussian noise (If working with the groundtruth)

            if self.use_gaussian_noise:
                mu = 0
                sigma = 0.05 
                
                x_offset, y_offset = np.random.normal(mu,sigma), np.random.normal(mu,sigma)

                bbox_object.x_corners += x_offset
                bbox_object.y_corners += y_offset
                
                bbox_object.x += x_offset
                bbox_object.y += y_offset

                theta = bbox_object.o # self.ego_orientation_cumulative_diff # Orientation angle (KITTI)
                # + math.pi/2 if using AB4COGT2SORT
                # + self.ego_orientation_cumulative_diff if using PointPillars
                beta = np.arctan2(bbox_object.x-self.ego_vehicle_x,self.ego_vehicle_y-bbox_object.y) # Observation angle (KITTI)

            # Calculate bounding box dimensions

            w = math.sqrt(pow(bbox_object.x_corners[3]-bbox_object.x_corners[1],2)+pow(bbox_object.y_corners[3]-bbox_object.y_corners[1],2))
            l = math.sqrt(pow(bbox_object.x_corners[0]-bbox_object.x_corners[1],2)+pow(bbox_object.y_corners[0]-bbox_object.y_corners[1],2))

            # Translate local to global coordinates

            aux_array = np.zeros((1,9))
            aux_array[0,4] = bbox_object.o
            aux_array[0,7:] = bbox_object.x, bbox_object.y

            current_pos = store_global_coordinates(self.tf_map2lidar,aux_array)

            if k == 0:
                bboxes = np.array([[current_pos[0,0],current_pos[1,0], 
                                    w, l,
                                    theta, beta,
                                    bbox_object.score,
                                    bbox_object.x,bbox_object.y]])
                
                types = np.array([bbox_object.type])
            else:
                bbox = np.array([[current_pos[0,0],current_pos[1,0], 
                                w, l,
                                theta, beta,
                                bbox_object.score,
                                bbox_object.x,bbox_object.y]])

                type_object = np.array([bbox_object.type])
                bboxes = np.concatenate([bboxes,bbox])
                types = np.concatenate([types,type_object])
            k += 1  
        bboxes = np.array(bboxes)

    return bboxes, types