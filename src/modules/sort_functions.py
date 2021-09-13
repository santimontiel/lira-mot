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
  Takes a bounding box in the form [x,y,w,l,theta,score] and returns z in the form
  [x,y,s,r,theta] where x,y is the centre of the box, s is the scale/area, r is
  the aspect ratio and theta is the bounding box angle
  """

  x = bbox[0]
  y = bbox[1]
  w = bbox[2]
  h = bbox[3]
  
  s = w*h         # Area of the rectangle
  r = w/float(h)  # Aspect ratio of the rectangle
  theta = bbox[4]

  return np.array([x,y,s,r,theta]).reshape((5,1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r,theta] and returns it in the form
    [x,y,w,l,theta] where x, y are the centroid, w and l the bounding box dimensions (in pixels)
    and theta is the bounding box angle
    """

    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    theta = x[4]

    if not score:
        return np.array([x[0],x[1],w,h,theta]).reshape((1,5))
    else:
        return np.array([x[0],x[1],w,h,theta,score]).reshape((1,6))

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
      dets - a numpy array of detections (x,y,w,l,theta,score), where x,y are the centroid coordinates (BEV plane), w and l the
      width and length of the obstacle (BEV plane), theta (rotation angle) and the detection score

      types - a numpy array with the corresponding type of the detections

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

def merged_bboxes_to_xywlthetascore_types(merged_bboxes):
  """
  merged_bboxes = [[x,y,z,l,w,h,theta,vel,type]]
  """

  bboxes = []
  types = []
  k = 0

  # Evaluate detections

  for merged_bbox in merged_bboxes:
    # Evaluate score ?
    l = merged_bbox[3]
    w = merged_bbox[4]

    if k == 0:
      bboxes = np.array([[merged_bbox[0],merged_bbox[1],
                          l,w,theta,vel]])
      types = np.array([merged_bbox[8]])
    else:
      bbox = np.array([[merged_bbox[0],merged_bbox[1],
                          l,w,theta,vel]])
      type_object = np.array([merged_bbox[8]])
      bboxes = np.concatenate([bboxes,bbox])
      types = np.concatenate([types,type_object])
    k += 1

  return bboxes,types



