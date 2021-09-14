#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 02:19:04 2021

@author: Carlos Gomez-Huelamo

Code to conduct tracking-by-detection, using a standard Kalman
Filter as State Estimation and Hungarian Algorithm as Data Association. Trajectory prediction
is carried out assuming a CTRV model (Constant Turn Rate and Velocity magnitude model)

Communications are based on ROS (Robot Operating Sytem)

Inputs: 

Outputs:  

Note that 

"""

import numpy as np
import math
import time

from . import geometric_functions
from . import sort_functions
from filterpy.kalman import KalmanFilter # Bayesian filters imports
from sklearn.preprocessing import PolynomialFeatures

# from scipy.optimize import linear_sum_assignment as linear_assignment # Hungarian Algorithm

def linear_assignment(cost_matrix):
  try:
    import lap
    print("Cost matrix: ", cost_matrix)
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    return matched_indices
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    matched_indices = np.array(list(zip(x, y)))
    return matched_indices

# Kalman

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  We assume constant velocity model
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    n_states = 9
    n_measurements = 5
    
    self.kf = KalmanFilter(dim_x=n_states, dim_z=n_measurements) # 9 variable vector, 5 measures
    
    # Transition matrix: x(k+1) = F*x(k)
    
    self.kf.F = np.array([[1,0,0,0,0,1,0,0,0],  # x
                          [0,1,0,0,0,0,1,0,0],  # y
                          [0,0,1,0,0,0,0,1,0],  # s
                          [0,0,0,1,0,0,0,0,0],  # r
                          [0,0,0,0,1,0,0,0,1],  # theta
                          [0,0,0,0,0,1,0,0,0],  # x'
                          [0,0,0,0,0,0,1,0,0],  # y'
                          [0,0,0,0,0,0,0,1,0],  # s'
                          [0,0,0,0,0,0,0,0,1]]) # theta'

    # Measurement matrix: z(k) = H*x(k)
    
    self.kf.H = np.array([[1,0,0,0,0,0,0,0,0],  # x
                          [0,1,0,0,0,0,0,0,0],  # y
                          [0,0,1,0,0,0,0,0,0],  # s
                          [0,0,0,1,0,0,0,0,0],  # r
                          [0,0,0,0,1,0,0,0,0]]) # theta

    # Measurement uncertainty/noise matrix
    
    self.kf.R[2:,2:] *= 10. # So s, r and theta are affected
    
    # Covariance matrix 
    
    self.kf.P[4:,4:] *= 10000. # Give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    
    # Process uncertainty/noise matrix 
    
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[-2,-2] *= 0.01
    self.kf.Q[5:,5:] *= 0.01

    # Filter state estimate matrix (Initial state)

    self.kf.x[:5] = sort_functions.convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1

    self.kf.update(sort_functions.convert_bbox_to_z(bbox)) 

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """ 
    if((self.kf.x[7]+self.kf.x[2])<=0):
      self.kf.x[7] *= 0.0
    
    self.kf.predict()
                      
    self.age += 1
    # if(self.time_since_update>0):
    #   self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(sort_functions.convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return sort_functions.convert_x_to_bbox(self.kf.x)

# Data association

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.001):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  print("Detections: ", detections)
  print("Trackers: ", trackers)
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,6),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  # Matched detections

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = geometric_functions.iou(det,trk)
      print("iou: ", iou_matrix[d,t])
  # print("iou matrix: ", iou_matrix)
  matched_indices = linear_assignment(-iou_matrix) # Hungarian Algorithm
  # print("Matched indices: ", matched_indices)

  # Unmatched detections
  
  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
      
  # Unmatched trackers
    
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # Filter out matched with low IOU
  
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)