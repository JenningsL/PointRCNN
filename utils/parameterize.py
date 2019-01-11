import numpy as np
import math

CENTER_SEARCH_RANGE = 3.8 # in meters
NUM_CENTER_BIN = 19 # must be odd
CENTER_BIN_SIZE = CENTER_SEARCH_RANGE / NUM_CENTER_BIN # in one direction
NUM_HEADING_BIN = 12

# Car, Pedestrian, Cyclist
type_mean_size = np.array([[3.88311640418,1.62856739989,1.52563191462],
    [0.84422524,0.66068622,1.76255119],
    [1.76282397,0.59706367,1.73698127]])

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

def obj_to_proposal_vec(obj, point):
    '''convert box3d related to a point to proposal vector'''
    # use point as origin
    center = obj.t - point
    # center_cls = np.zeros((NUM_CENTER_BIN*NUM_CENTER_BIN,))
    center_res = np.zeros((2,))
    c_x = center[0] / CENTER_BIN_SIZE + NUM_CENTER_BIN / 2
    c_z = center[2] / CENTER_BIN_SIZE + NUM_CENTER_BIN / 2
    c_x = max(min(NUM_CENTER_BIN-1, c_x), 0)
    c_z = max(min(NUM_CENTER_BIN-1, c_z), 0)
    # center_cls[int(c_z*NUM_CENTER_BIN+c_x)] = 1
    center_cls = int(c_z*NUM_CENTER_BIN+c_x)
    bin_center = np.array([
        (c_x - NUM_CENTER_BIN / 2) * CENTER_BIN_SIZE,
        0,
        (c_z - NUM_CENTER_BIN / 2) * CENTER_BIN_SIZE
    ])
    center_res = center - bin_center
    # true center = point + bin_center + center_res
    # print(obj.t, point + bin_center + center_res)
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
    center_cls, center_res, angle_cls,angle_res, size_cls, size_res = obj_to_proposal_vec(obj, np.array([-27.1,1.3,4.3]))
    print(center_cls, center_res, angle_cls,angle_res, size_cls, size_res)
