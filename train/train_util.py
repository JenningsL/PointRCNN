from __future__ import print_function

#import cPickle as pickle
import pickle
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
from parameterize import class2angle, class2size, class2center, NUM_HEADING_BIN
from box_util import box3d_iou

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def compute_box3d_iou(points, indices, heading_logits, heading_residuals,
        size_logits, size_residuals,center_x_logits,center_z_logits,center_x_residuals,
        center_y_residuals,center_z_residuals,
        heading_cls_label, heading_res_label, size_cls_label, size_res_label,
        center_x_cls_label, center_z_cls_label,
        center_x_res_label, center_y_res_label, center_z_res_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        end_points,
        labels
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    fg_points = heading_logits.shape[1]
    heading_class = np.argmax(heading_logits, 2) # B,M
    size_class = np.argmax(size_logits, 2) # B,M
    center_x_cls = np.argmax(center_x_logits, 2)
    center_z_cls = np.argmax(center_z_logits, 2)
    center_cls = np.stack([center_x_cls, center_z_cls], axis=-1)
    #center_res = np.stack([center_x_residuals,center_y_residuals,center_z_residuals], axis=-1)

    center_cls_label = np.stack([center_x_cls_label, center_z_cls_label], axis=-1)

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        for j in range(fg_points):
            heading_angle = class2angle(heading_class[i,j],
                heading_residuals[i,j, heading_class[i,j]], NUM_HEADING_BIN)
            #print('size_class: ', size_class[i,j])
            #print('size_residuals: ', size_residuals[i,j,size_class[i,j]])
            box_size = class2size(size_class[i,j], size_residuals[i,j,size_class[i,j]])
            #print('box_size', box_size)
            center_res = np.array([
                center_x_residuals[i,j,center_x_cls[i,j]],
                center_y_residuals[i,j],
                center_z_residuals[i,j,center_z_cls[i,j]]
            ])
            #print('center_cls:', center_cls[i,j])
            #print('center_res:', center_res)
            center_pred = class2center(center_cls[i,j], center_res, points[i,j])
            #print('center_pred:', center_pred)
            corners_3d = get_3d_box(box_size, heading_angle, center_pred)
            #print('-------------------------------------')

            # ground truth
            fg_idx = indices[i,j,1] # which point is sampled
            #print('fg_idx', fg_idx)
            heading_angle_label = class2angle(heading_cls_label[i,fg_idx],
                heading_res_label[i,fg_idx], NUM_HEADING_BIN)
            #print('size_cls_label: ', size_cls_label[i,fg_idx])
            #print('size_res_label: ', size_res_label[i,fg_idx])
            box_size_label = class2size(size_cls_label[i,fg_idx], size_res_label[i,fg_idx])
            #print('box_size_label: ', box_size_label)
            center_res_label = np.array([
                center_x_res_label[i,fg_idx],
                center_y_res_label[i,fg_idx],
                center_z_res_label[i,fg_idx]
            ])
            #print('center_cls_label: ', center_cls_label[i,fg_idx])
            #print('center_res_label: ', center_res_label)
            center_label = class2center(center_cls_label[i,fg_idx], center_res_label, points[i,j])
            #print('center_label: ', center_label)
            corners_3d_label = get_3d_box(box_size_label,
                heading_angle_label, center_label)
            #print(corners_3d.shape, corners_3d_label.shape)
            iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
            iou3d_list.append(iou_3d)
            iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)

