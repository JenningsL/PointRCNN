import numpy as np
import math

CENTER_SEARCH_RANGE = 3.0 # in meters
NUM_CENTER_BIN = 12
CENTER_BIN_SIZE = CENTER_SEARCH_RANGE * 2 / NUM_CENTER_BIN # in one direction
NUM_HEADING_BIN = 12

# Car, Pedestrian, Cyclist
type_mean_size = np.array([[3.88311640418,1.62856739989,1.52563191462],
    [0.84422524,0.66068622,1.76255119],
    [1.76282397,0.59706367,1.73698127]])
NUM_SIZE_CLUSTER = type_mean_size.shape[0] # one cluster for each type

def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.
    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - \
        (class_id * angle_per_class + angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle

def size2class(size):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    dist = np.linalg.norm(type_mean_size - size, axis=1)
    size_class = np.argmin(dist)
    size_residual = size - type_mean_size[size_class]
    return size_class, size_residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = type_mean_size[pred_cls]
    return mean_size + residual

def center2class(obj_center, point):
    center = obj_center - point
    bin_x = int((obj_center[0] - point[0] + CENTER_SEARCH_RANGE) / CENTER_BIN_SIZE)
    bin_z = int((obj_center[2] - point[2] + CENTER_SEARCH_RANGE) / CENTER_BIN_SIZE)
    # limit the bin class label range
    bin_x = np.clip(bin_x, 0, NUM_CENTER_BIN-1)
    bin_z = np.clip(bin_z, 0, NUM_CENTER_BIN-1)
    center_cls = np.array([bin_x, bin_z])
    center_res = np.array([
        1/CENTER_BIN_SIZE * (obj_center[0] - point[0] + CENTER_SEARCH_RANGE - (bin_x+0.5)*CENTER_BIN_SIZE),
        obj_center[1] - point[1],
        1/CENTER_BIN_SIZE * (obj_center[2] - point[2] + CENTER_SEARCH_RANGE - (bin_z+0.5)*CENTER_BIN_SIZE),
    ])
    return center_cls, center_res

def class2center(center_cls, center_res, point):
    # recover true center from center_cls and center_res
    bin_center = np.array([
        center_cls[0] * CENTER_BIN_SIZE + CENTER_BIN_SIZE/2 - CENTER_SEARCH_RANGE + point[0],
        point[1],
        center_cls[1] * CENTER_BIN_SIZE + CENTER_BIN_SIZE/2 - CENTER_SEARCH_RANGE + point[2]
    ])
    obj_center = bin_center + np.array([center_res[0]*CENTER_BIN_SIZE, center_res[1], center_res[2]*CENTER_BIN_SIZE])
    return obj_center

def obj_to_proposal_vec(obj, point):
    '''convert box3d related to a point to proposal vector'''
    # use point as origin
    obj_center = obj.t
    center_cls, center_res = center2class(obj_center, point)
    ## encode heading
    angle_cls, angle_res = angle2class(obj.ry, NUM_HEADING_BIN)
    # print(angle_cls, angle_res)
    size_cls, size_res = size2class(np.array([obj.l,obj.w,obj.h]))
    return center_cls, center_res, angle_cls,angle_res, size_cls, size_res

if __name__ == '__main__':
    class Box(object):
        pass
    obj = Box()
    obj.t = np.array([-28.3,1.2,5.2])
    obj.ry = math.pi * 0.3
    obj.l = 1
    obj.w = 0.6
    obj.h = 0.7
    point = np.array([-27.1,1.3,4.3])
    center_cls, center_res, angle_cls,angle_res, size_cls, size_res = obj_to_proposal_vec(obj, point)
    print(center_cls, center_res, angle_cls,angle_res, size_cls, size_res)
    print(class2center(center_cls, center_res, point))
    print(class2size(size_cls, size_res))
    print(class2angle(angle_cls,angle_res, NUM_HEADING_BIN))
