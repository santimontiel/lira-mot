# Math and geometry imports

import math
import numpy 
import cv2

from shapely.geometry import Polygon

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
    nq = numpy.dot(q, q)
    if nq < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def euclidean_distance(pose_a,pose_b):
    """
    """
    xa = pose_a.pose.pose.position.x
    ya = pose_a.pose.pose.position.y
    za = pose_a.pose.pose.position.z

    xb = pose_b.pose.pose.position.x
    yb = pose_b.pose.pose.position.y
    zb = pose_b.pose.pose.position.z

    distance = math.sqrt(pow(xa-xb,2)+pow(ya-yb,2)+pow(za-zb,2))

    return distance

def rotz(t):
    """ 
    Rotation about the z-axis
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                     [s,   c,  0],
                     [0,   0,  1]])

def compute_corners(bbox):
    """
    Compute the corners of the rectangle given its x,y centroid, width and length and
    its rotation (clockwise if you see the screen, according to OpenCV)
    """
    
    rotation = bbox[4]
    
    if rotation > math.pi:
        rotation = rotation - math.pi
    
    R = rotz(rotation)

    # 3D bounding box corners

    x, y, w, l = bbox[0], bbox[1], bbox[2], bbox[3]

    x_corners = [-l/2,-l/2,l/2,l/2]
    y_corners = [w/2,-w/2,w/2,-w/2]
    z_corners = [0,0,0,0]

    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d = np.dot(R, corners_3d)[0:2]
    corners_3d = corners_3d + np.vstack([x,y])
    
    corners = []
    for i in range(4):
        c = int(round(corners_3d[0,i].item())), int(round(corners_3d[1,i].item()))
        corners.append(c)
    corners = tuple(corners)
    return corners

def iou(bb_1,bb_2): 
  """
  Computes IOU between two (possibly) rotated bounding boxes in the form [x,y,w,l,theta]
  """

  corners_1 = compute_corners(bb_1)
  corners_2 = compute_corners(bb_2)

  # To build the polygon -> Left-bottom corner, Right-bottom, Top-right corner, Top-left corner

  b1 = Polygon([corners_1[2],corners_1[3],corners_1[1],corners_1[0]]) 
  b2 = Polygon([corners_2[2],corners_2[3],corners_2[1],corners_2[0]])
  
  intersection = b1.intersection(b2).area
  union = b1.union(b2).area

  if union > 0.0:
    o = intersection / union
    return(o)
  else:
    return 0.0



