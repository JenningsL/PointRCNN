from __future__ import print_function
import os
import sys
import cPickle as pickle
import numpy as np
from shapely.geometry import Polygon
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from kitti_object import *
import kitti_util as utils
from data_conf import type_whitelist, difficulties_whitelist
from data_util import extract_pc_in_box3d, ProposalObject
from nms_rotate import nms_rotate_cpu

def find_match_label(prop_corners, labels_corners):
    '''
    Find label with largest IOU. Label boxes can be rotated in xy plane
    '''
    # labels = MultiPolygon(labels_corners)
    labels = map(lambda corners: Polygon(corners), labels_corners)
    target = Polygon(prop_corners)
    largest_iou = 0
    largest_idx = -1
    for i, label in enumerate(labels):
        area1 = label.area
        area2 = target.area
        intersection = target.intersection(label).area
        iou = intersection / (area1 + area2 - intersection)
        if iou > largest_iou:
            largest_iou = iou
            largest_idx = i
    return largest_idx, largest_iou

proposal_cache = {}
def get_proposals(rpn_out, nms_thres, max_keep):
    cache_key = '{0}_{1}_{2}'.format(rpn_out['frame_id'], nms_thres, max_keep)
    if cache_key in proposal_cache:
        return proposal_cache[cache_key]
    proposals = []
    bev_boxes = []
    for ry, center, size in zip(rpn_out['angles'], rpn_out['centers'], rpn_out['sizes']):
        bev_boxes.append([center[0], center[2], size[0], size[2], 180*ry/np.pi])
    bev_boxes = np.array(bev_boxes)
    nms_idx = nms_rotate_cpu(bev_boxes, rpn_out['scores'], nms_thres, max_keep)
    for ind in nms_idx:
        # to ProposalObject
        x,y,z = rpn_out['centers'][ind]
        l, h, w = rpn_out['sizes'][ind]
        ry = rpn_out['angles'][ind]
        proposal = ProposalObject(np.array([x,y,z,l, h, w, ry]))
        proposals.append(proposal)
    proposal_cache[cache_key] = proposals
    return proposals

def stat(kitti_path, split, data_dir, type_whitelist, difficulties_whitelist):
    kitti_dataset = kitti_object(kitti_path, 'training')
    with open(os.path.join(kitti_path, split + '.txt')) as f:
        frame_ids = [line.rstrip('\n') for line in f]

    missed_sample = 0
    missed_seg = 0
    missed_seg_sample = 0
    missed_obj_point = 0
    tp = 0
    fp = 0
    fn = 0
    total_recall = {10: 0, 20: 0, 30: 0, 40: 0, 50: 0, 100: 0, 300: 0}
    label_count = 0
    for frame_id in frame_ids:
        print(frame_id)
        sys.stdout.flush()
        try:
            with open(os.path.join(data_dir, frame_id+'.pkl'), 'rb') as fin:
                rpn_out = pickle.load(fin)
        except Exception as e:
            print(e)
            continue
        data_idx = int(frame_id)
        pc_choices = rpn_out['pc_choices']
        seg_pred_point = rpn_out['segmentation'] # point seg
        seg_pred = rpn_out['segmentation_fuse'] # point+img seg
        fg_indices = rpn_out['fg_indices'] # after sampling


        seg_pred = seg_pred_point

        pc_velo = kitti_dataset.get_lidar(data_idx)
        image = kitti_dataset.get_image(data_idx)
        calib = kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        img_height, img_width = image.shape[0:2]
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)
        pc_velo = pc_velo[img_fov_inds, :3]
        pc_velo_sampled = pc_velo[pc_choices]
        pc_rect = calib.project_velo_to_rect(pc_velo)
        pc_rect_sampled = calib.project_velo_to_rect(pc_velo_sampled) # 16384
        pc_seg = pc_rect_sampled[seg_pred>0] # all seg fg points
        if pc_seg.shape[0] > 0:
            pc_seg_sampled = pc_rect_sampled[fg_indices] # sampled fg points 2048
        else:
            pc_seg_sampled = pc_seg
        objects = kitti_dataset.get_label_objects(data_idx)
        objects = filter(lambda obj: obj.type in type_whitelist and obj.difficulty in difficulties_whitelist, objects)
        gt_boxes = [] # ground truth boxes
        gt_boxes_xy = []
        seg_mask = np.zeros((pc_rect_sampled.shape[0],))
        for obj in objects:
            _,obj_box_3d = utils.compute_box_3d(obj, calib.P)
            # skip label with no point
            _,obj_mask = extract_pc_in_box3d(pc_rect, obj_box_3d)
            if np.sum(obj_mask) == 0:
                continue
            gt_boxes.append(obj_box_3d)
            gt_boxes_xy.append(obj_box_3d[:4, [0,2]])
            label_count += 1
            _, obj_mask = extract_pc_in_box3d(pc_rect_sampled, obj_box_3d)
            seg_mask[obj_mask] = 1
            if np.sum(obj_mask) == 0:
                missed_sample += 1
            _, mask = extract_pc_in_box3d(pc_seg, obj_box_3d)
            if np.sum(mask) == 0:
                missed_seg += 1
                missed_obj_point += np.sum(obj_mask)
            _, mask = extract_pc_in_box3d(pc_seg_sampled, obj_box_3d)
            if np.sum(mask) == 0:
                missed_seg_sample += 1

        tp += np.sum(np.logical_and(seg_pred == seg_mask, seg_mask != 0))
        fp += np.sum(np.logical_and(seg_pred != seg_mask, seg_mask == 0))
        fn += np.sum(np.logical_and(seg_pred != seg_mask, seg_mask != 0))
        for max_keep in total_recall.keys():
            recall = np.zeros((len(objects),))
            proposals = get_proposals(rpn_out, 0.7, max_keep)
            for prop in proposals:
                if np.sum(recall) == len(objects):
                    break
                b2d,prop_box_3d = utils.compute_box_3d(prop, calib.P)
                prop_box_xy = prop_box_3d[:4, [0,2]]
                gt_idx, gt_iou = find_match_label(prop_box_xy, gt_boxes_xy)
                if gt_idx == -1:
                    continue
                if gt_iou >= 0.5:
                    recall[gt_idx] = 1
                #if objects[gt_idx].type != 'Car' and gt_iou >= 0.5:
                #    print(objects[gt_idx].type)
            total_recall[max_keep] += np.sum(recall)

    print('Missed sample ratio: ', float(missed_sample) / label_count)
    print('Missed seg ratio: ', float(missed_seg) / label_count)
    print('Missed object average point: ', missed_obj_point / missed_seg)
    print('Missed seg sample ratio: ', float(missed_seg_sample) / label_count)
    for max_keep in total_recall.keys():
        total_recall[max_keep] = float(total_recall[max_keep]) / label_count
    print('Recall on proposal num: ', total_recall)
    print('Seg recall: ', float(tp)/(tp+fn))
    print('Seg precision: ', float(tp)/(tp+fp))
    sys.stdout.flush()

if __name__ == '__main__':
    for label_type in type_whitelist:
        print('============================> {}'.format(label_type))
        sys.stdout.flush()
        stat(sys.argv[1], sys.argv[2], sys.argv[3], [label_type], [1])
    print('============================> {}'.format('All'))
    stat(sys.argv[1], sys.argv[2], sys.argv[3], type_whitelist, difficulties_whitelist)

