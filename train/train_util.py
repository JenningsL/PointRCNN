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
from box_util import box3d_iou, get_3d_box

# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def compute_box3d_iou(pred_boxes, gt_boxes, nms_indices):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        pred_boxes,
        gt_boxes
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    iou2d_list = []
    iou3d_list = []
    for i in range(len(pred_boxes)):
        ind = nms_indices[i]
        ind = ind[ind!=-1]
        for corners_3d, corners_3d_label in zip(pred_boxes[i,ind], gt_boxes[i,ind]):
            iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
            iou3d_list.append(iou_3d)
            iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)

def compute_proposal_recall(batch_pred_boxes, batch_gt_boxes, nms_indices, iou_threshold=0.5):
    '''
    batch_pred_boxes: (B,FG_POINT_NUM,8,3)
    batch_gt_boxes: (B,?,8,3)
    '''
    total_recall = 0
    total_labels = 0
    for pred_boxes, gt_boxes, ind in zip(batch_pred_boxes, batch_gt_boxes, nms_indices):
        ind = ind[ind!=-1]
        pred_boxes = pred_boxes[ind] # pred after nms
        recall = np.zeros((len(gt_boxes),))
        for i, gt_box in enumerate(gt_boxes):
            for pred_box in pred_boxes:
                iou_3d, iou_2d = box3d_iou(pred_box, gt_box)
                if iou_2d > iou_threshold:
                    recall[i] = 1
                    break
        total_recall += np.sum(recall)
        total_labels += len(gt_boxes)
    return float(total_recall) / total_labels

if __name__ == '__main__':
    gt_boxes = np.array([[
        [[-0.2665488 ,  1.68      , 21.26691373],[ 1.39309225,  1.68      , 21.23239432],[ 1.3265488 ,  1.68      , 18.03308627],[-0.33309225,  1.68      , 18.06760568],[-0.2665488 ,  0.07      , 21.26691373],[ 1.39309225,  0.07      , 21.23239432],[ 1.3265488 ,  0.07      , 18.03308627],[-0.33309225,  0.07      , 18.06760568]]
    ]])
    prop_boxes = np.array([[
        [[-0.2665488 ,  1.68      , 21.26691373],[ 1.39309225,  1.68      , 21.23239432],[ 1.3265488 ,  1.68      , 18.03308627],[-0.33309225,  1.68      , 18.06760568],[-0.2665488 ,  0.07      , 21.26691373],[ 1.39309225,  0.07      , 21.23239432],[ 1.3265488 ,  0.07      , 18.03308627],[-0.33309225,  0.07      , 18.06760568]]
    ]])
    nms_inds = np.array([[0]])
    print(compute_proposal_recall(prop_boxes, gt_boxes, nms_inds))
    print(compute_box3d_iou(prop_boxes, gt_boxes, nms_inds))
