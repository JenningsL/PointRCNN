from __future__ import print_function

import os
import sys
import numpy as np
import copy
import random
import threading
from Queue import Queue
import time
import math
import cPickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'visualize/mayavi'))
from nms_rotate import nms_rotate_cpu
from kitti_object import *
import kitti_util as utils
from box_encoder import BoxEncoder
from data_util import rotate_points_along_y, shift_point_cloud, extract_pc_in_box3d
from data_util import ProposalObject, np_read_lines, find_match_label, random_shift_box3d
from data_conf import type_whitelist, difficulties_whitelist

g_type2onehotclass = {'NonObject': 0, 'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
NUM_HEADING_BIN = 9
NUM_CENTER_BIN = 6
CENTER_SEARCH_RANGE = 1.5
HEADING_SEARCH_RANGE = 0.25*np.pi

box_encoder = BoxEncoder(CENTER_SEARCH_RANGE, NUM_CENTER_BIN, HEADING_SEARCH_RANGE, NUM_HEADING_BIN)

class Dataset(object):
    def __init__(self, npoints, kitti_path, split, is_training, \
        types=type_whitelist, difficulties=difficulties_whitelist):
        self.npoints = npoints
        self.kitti_path = kitti_path
        self.data_dir = './rcnn_data_'+split
        #self.batch_size = batch_size
        self.split = split
        self.is_training = is_training
        if split in ['train', 'val']:
            self.kitti_dataset = kitti_object(kitti_path, 'training')
            self.frame_ids = self.load_split_ids(split)
        else:
            self.kitti_dataset = kitti_object_video(
                os.path.join(kitti_path, 'image_02/data'),
                os.path.join(kitti_path, 'velodyne_points/data'),
                kitti_path)
            self.frame_ids = range(self.kitti_dataset.num_samples)
        random.shuffle(self.frame_ids)
        self.num_channel = 6 # xyz intensity is_obj_one_hot

        self.types_list = types
        self.difficulties_list = difficulties

        self.sample_id_counter = -1 # as id for sample
        self.last_sample_id = None
        # preloading
        self.stop = False
        self.data_buffer = Queue(maxsize=1024)

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def load(self, aug=False):
        # load proposals
        i = 0
        last_sample_id = None
        while not self.stop:
            frame_id = self.frame_ids[i]
            #print('loading ' + frame_id)
            frame_data = {}
            samples = \
                self.load_frame_data(frame_id, aug)
            for s in samples:
                s['frame_id'] = frame_id
                self.data_buffer.put(s)
            if len(samples) > 0:
                last_sample_id = samples[-1]['id']
            if i == len(self.frame_ids) - 1:
                self.last_sample_id = last_sample_id
                random.shuffle(self.frame_ids)
            i = (i + 1) % len(self.frame_ids)

    def stop_loading(self):
        self.stop = True
        while not self.data_buffer.empty():
            item = self.data_buffer.get()
            self.data_buffer.task_done()

    def get_next_batch(self, bsize, need_id=False):
        is_last_batch = False

        batch = {
            'ids': [],
            'pointcloud': np.zeros((bsize, self.npoints, self.num_channel), dtype=np.float32),
            'images': np.zeros((bsize, 360, 1200, 3), dtype=np.float32),
            'img_seg_map': np.zeros((bsize, 360, 1200, 4), dtype=np.float32),
            'calib': np.zeros((bsize, 3, 4), dtype=np.float32),
            'label': np.zeros((bsize,), dtype=np.int32),
            'prop_box': np.zeros((bsize, 7), dtype=np.float32),
            'center_x_cls': np.zeros((bsize,), dtype=np.int32),
            'center_z_cls': np.zeros((bsize,), dtype=np.int32),
            'center_x_res': np.zeros((bsize,), dtype=np.float32),
            'center_y_res': np.zeros((bsize,), dtype=np.float32),
            'center_z_res': np.zeros((bsize,), dtype=np.float32),
            'angle_cls': np.zeros((bsize,), dtype=np.int32),
            'size_cls': np.zeros((bsize,), dtype=np.int32),
            'angle_res': np.zeros((bsize,), dtype=np.float32),
            'size_res': np.zeros((bsize, 3), dtype=np.float32),
            'gt_box_of_prop': np.zeros((bsize, 8, 3), dtype=np.float32),
            'train_regression': np.zeros((bsize,), dtype=bool)
        }
        for i in range(bsize):
            sample = self.data_buffer.get()
            if sample['id'] == self.last_sample_id:
                is_last_batch = True
                self.last_sample_id = None
            batch['ids'].append(sample['frame_id'])
            choice = np.random.choice(sample['pointcloud'].shape[0], self.npoints, replace=True)
            batch['pointcloud'][i,...] = sample['pointcloud'][choice]
            batch['calib'][i,...] = sample['calib']
            batch['images'][i,...] = sample['image']
            batch['img_seg_map'][i,...] = sample['img_seg_map']
            batch['label'][i] = sample['class']
            batch['prop_box'][i,...] = sample['proposal_box']
            batch['center_x_cls'][i] = sample['center_cls'][0]
            batch['center_z_cls'][i] = sample['center_cls'][1]
            batch['center_x_res'][i] = sample['center_res'][0]
            batch['center_y_res'][i] = sample['center_res'][1]
            batch['center_z_res'][i] = sample['center_res'][2]
            batch['angle_cls'][i] = sample['angle_cls']
            batch['size_cls'][i] = sample['size_cls']
            batch['angle_res'][i] = sample['angle_res']
            batch['size_res'][i,...] = sample['size_res']
            batch['gt_box_of_prop'][i,...] = sample['gt_box']
            batch['train_regression'][i] = sample['train_regression']

        return batch, is_last_batch

    def viz_frame(self, pc_rect, mask, gt_boxes, proposals):
        import mayavi.mlab as mlab
        from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
        fig = draw_lidar(pc_rect)
        fig = draw_lidar(pc_rect[mask==1], fig=fig, pts_color=(1, 1, 1))
        fig = draw_gt_boxes3d(gt_boxes, fig, draw_text=False, color=(1, 1, 1))
        fig = draw_gt_boxes3d(proposals, fig, draw_text=False, color=(1, 0, 0))
        raw_input()

    def viz_sample(self, sample):
        import mayavi.mlab as mlab
        from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
        pc_rect = sample['pointcloud']
        proposal = ProposalObject(sample['proposal_box'])
        proposal.ry = 0
        proposal.t = np.zeros((3,))
        _, prop_box = utils.compute_box_3d(proposal, sample['calib'])
        fig = draw_lidar(pc_rect)
        mask = pc_rect[:,5] == 1
        fig = draw_lidar(pc_rect[mask], fig=fig, pts_color=(1, 1, 1))
        fig = draw_gt_boxes3d([prop_box], fig, draw_text=False, color=(1, 1, 1))
        raw_input()

    def get_proposals_gt(self, obj):
        center = obj.t + np.random.normal(0, 0.1, 3)
        ry = obj.ry + np.random.normal(0, np.pi/8, 1)
        # ensure that ry is [-pi, pi]
        if obj.ry > np.pi:
            obj.ry -= 2*np.pi
        elif obj.ry < -np.pi:
            obj.ry += 2*np.pi
        # ry = obj.ry
        l = obj.l + np.random.normal(0, 0.1, 1)[0]
        h = obj.h + np.random.normal(0, 0.1, 1)[0]
        w = obj.w + np.random.normal(0, 0.1, 1)[0]

        return ProposalObject(np.array([center[0],center[1],center[2],l, h, w, ry]))

    def get_proposals(self, rpn_out):
        proposals = []
        if self.split == 'train':
            nms_thres = 0.85
            max_keep = 300
        else:
            nms_thres = 0.8
            max_keep = 100
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
        return proposals

    def fill_proposals_with_gt(self, objects):
        gt_proposals = []
        for obj in objects:
            gt_proposals.append(self.get_proposals_gt(obj))
        return gt_proposals

    def load_frame_data(self, data_idx_str, aug):
        '''load one frame'''
        start = time.time()
        data_idx = int(data_idx_str)

        try:
            with open(os.path.join(self.data_dir, data_idx_str+'.pkl'), 'rb') as fin:
                rpn_out = pickle.load(fin)
            # load image segmentation output
            img_seg_map = np.load(os.path.join(self.data_dir, data_idx_str+'_seg.npy'))
        except Exception as e:
            print(e)
            return []
        proposals = self.get_proposals(rpn_out)
        #print(data_idx_str)
        calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        if self.is_training:
            objects = self.kitti_dataset.get_label_objects(data_idx)
        else:
            # while testing, all proposals will have class 0
            objects = []
        image = self.kitti_dataset.get_image(data_idx)
        pc_velo = self.kitti_dataset.get_lidar(data_idx)
        img_height, img_width = image.shape[0:2]
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)
        pc_velo = pc_velo[img_fov_inds, :]
        # Same point sampling as RPN
        choice = rpn_out['pc_choices']
        point_set = pc_velo[choice, :]
        # point_set = pc_velo
        pc_rect = np.zeros_like(point_set)
        pc_rect[:,0:3] = calib.project_velo_to_rect(point_set[:,0:3])
        pc_rect[:,3] = point_set[:,3]
        # Segmentation results from RPN
        seg_one_hot = np.zeros((pc_rect.shape[0], 2))
        bg_ind = rpn_out['segmentation'] == 0
        fg_ind = rpn_out['segmentation'] == 1
        seg_one_hot[bg_ind,0] = 1
        seg_one_hot[fg_ind,1] = 1
        pc_rect = np.concatenate((pc_rect, seg_one_hot), axis=-1) # 6 channels
        objects = filter(lambda obj: obj.type in self.types_list and obj.difficulty in self.difficulties_list, objects)
        gt_boxes = [] # ground truth boxes
        gt_boxes_xy = []
        recall = np.zeros((len(objects),), dtype=np.int32)
        for obj in objects:
            _,obj_box_3d = utils.compute_box_3d(obj, calib.P)
            # skip label with no point
            _,obj_mask = extract_pc_in_box3d(pc_rect, obj_box_3d)
            if np.sum(obj_mask) == 0:
                continue
            gt_boxes.append(obj_box_3d)
            gt_boxes_xy.append(obj_box_3d[:4, [0,2]])
        positive_samples = []
        negative_samples = []
        show_boxes = []
        # boxes_2d = []
        def process_proposal(prop):
            b2d,prop_box_3d = utils.compute_box_3d(prop, calib.P)
            prop_box_xy = prop_box_3d[:4, [0,2]]
            max_idx, max_iou = find_match_label(prop_box_xy, gt_boxes_xy)
            sample = self.get_sample(pc_rect, image, img_seg_map, calib, prop, max_iou, max_idx, objects)
            # print(max_iou)
            if not sample:
                return -1
            if sample['class'] == 0:
                show_boxes.append(prop_box_3d)
                negative_samples.append(sample)
            else:
                positive_samples.append(sample)
                show_boxes.append(prop_box_3d)
                recall[max_idx] = 1
            return sample['class']
        aug_proposals = []
        AUG_X = {1:1, 2:2, 3:2}
        for prop in proposals:
            cls = process_proposal(prop)
            if not aug or cls <= 0:
                continue
            for x in range(AUG_X[cls]):
                prop_ = random_shift_box3d(copy.deepcopy(prop), 0.1)
                aug_proposals.append(prop_)
        # add more proposals using label to increase training samples
        '''
        if self.split == 'train':
            miss_objs = [objects[i] for i in range(len(objects)) if recall[i]==0]
            aug_proposals += self.fill_proposals_with_gt(miss_objs)
        '''
        for prop in aug_proposals:
            process_proposal(prop)
        if self.is_training:
            random.shuffle(negative_samples)
            samples = positive_samples + negative_samples[:len(positive_samples)]
        else:
            samples = positive_samples + negative_samples
        random.shuffle(samples)
        # self.viz_frame(pc_rect, np.zeros((pc_rect.shape[0],)), gt_boxes, show_boxes)
        return samples

    def get_sample(self, pc_rect, image, img_seg_map, calib, proposal_, max_iou, max_idx, objects):
        thres_low = 0.3
        thres_high = 0.55
        if max_iou >= thres_high:
            label = objects[max_idx]
        if max_iou < thres_low:
            label = None
        if self.is_training and max_iou >= thres_low and max_iou < thres_high:
            return False
        # expand proposal boxes
        proposal_expand = copy.deepcopy(proposal_)
        proposal_expand.l += 0.5
        proposal_expand.w += 0.5
        _, box_3d = utils.compute_box_3d(proposal_expand, calib.P)
        _, mask = extract_pc_in_box3d(pc_rect, box_3d)
        # ignore proposal with no points
        if(np.sum(mask) == 0):
            return False

        points = pc_rect[mask,:]
        points_with_feats = np.zeros((points.shape[0], self.num_channel))
        points_with_feats[:,:6] = points # xyz, intensity, seg_one_hot
        # pooled points canonical transformation
        points_with_feats[:,:3] -= proposal_.t
        points_with_feats[:,:3] = rotate_points_along_y(points_with_feats[:,:3], proposal_.ry)

        sample = {}
        self.sample_id_counter += 1
        sample['id'] = self.sample_id_counter
        sample['class'] = 0
        sample['pointcloud'] = points_with_feats
        sample['image'] = cv2.resize(image, (1200, 360))
        sample['img_seg_map'] = img_seg_map
        sample['calib'] = np.copy(calib.P)
        # scale projection matrix
        sample['calib'][0,:] *= (1200.0 / image.shape[1])
        sample['calib'][1,:] *= (360.0 / image.shape[0])
        sample['proposal_box'] = np.array([proposal_.t[0], proposal_.t[1], proposal_.t[2],
            proposal_.l, proposal_.h, proposal_.w, proposal_.ry])
        sample['center_cls'] = np.zeros((2,), dtype=np.int32)
        sample['center_res'] = np.zeros((3,))
        sample['angle_cls'] = 0
        sample['angle_res'] = 0
        sample['size_cls'] = 0
        sample['size_res'] = np.zeros((3,))
        sample['gt_box'] = np.zeros((8,3))
        sample['train_regression'] = max_iou >= thres_high
        if label:
            sample['class'] = g_type2onehotclass[label.type]
            # rotation canonical transformation
            label_norm = copy.deepcopy(label)
            label_norm.ry = label.ry - proposal_.ry
            obj_vec = box_encoder.encode(label_norm, proposal_.t)
            sample['center_cls'] = obj_vec[0]
            sample['center_res'] = obj_vec[1]
            sample['angle_cls'] = obj_vec[2]
            sample['angle_res'] = obj_vec[3]
            sample['size_cls'] = obj_vec[4]
            sample['size_res'] = obj_vec[5]
            _, gt_box_3d = utils.compute_box_3d(label, calib.P)
            sample['gt_box'] = gt_box_3d
            #self.viz_sample(sample)
        return sample

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    split = sys.argv[2]

    sys.path.append('models')
    from collections import namedtuple
    import tensorflow as tf
    from img_vgg_pyramid import ImgVggPyr
    import projection
    VGG_config = namedtuple('VGG_config', 'vgg_conv1 vgg_conv2 vgg_conv3 vgg_conv4 l2_weight_decay')

    dataset = Dataset(512, kitti_path, split, True)
    dataset.load(True)
    produce_thread = threading.Thread(target=dataset.load, args=(True,))
    produce_thread.start()
    i = 0
    total = 0
    while(True):
        batch_data, is_last_batch = dataset.get_next_batch(1, need_id=True)
        # total += np.sum(batch_data[1] == 1)
        # print('foreground points:', np.sum(batch_data[1] == 1))
        print(batch_data['ids'])
        with tf.Session() as sess:
            img_vgg = ImgVggPyr(VGG_config(**{
                'vgg_conv1': [2, 32],
                'vgg_conv2': [2, 64],
                'vgg_conv3': [3, 128],
                'vgg_conv4': [3, 256],
                'l2_weight_decay': 0.0005
            }))
            img_pixel_size = np.asarray([360, 1200])
            img_preprocessed = img_vgg.preprocess_input(batch_data['images'], img_pixel_size)
            box2d_corners, box2d_corners_norm = projection.tf_project_to_image_space(
                batch_data['prop_box'],
                batch_data['calib'], img_pixel_size)

            box2d_corners_norm_reorder = tf.stack([
                tf.gather(box2d_corners_norm, 1, axis=-1),
                tf.gather(box2d_corners_norm, 0, axis=-1),
                tf.gather(box2d_corners_norm, 3, axis=-1),
                tf.gather(box2d_corners_norm, 2, axis=-1),
            ], axis=-1)
            img_rois = tf.image.crop_and_resize(
                img_preprocessed,
                box2d_corners_norm_reorder, # reorder
                tf.range(0, 1),
                [100,100])
            corners, corners_norm = sess.run([box2d_corners,box2d_corners_norm])
            # break
            whole_img = cv2.resize(batch_data['images'][0]/255, (1200,360))
            # corner = corners[0].astype(int)
            corner = (corners_norm[0] * np.array([1200,360,1200,360])).astype(int)
            print('proposal box: ', batch_data['prop_box'][0])
            print('projection matrix: ', batch_data['calib'][0])
            print(corner, corners[0].astype(int))
            cv2.rectangle(whole_img,(corner[0], corner[1]),(corner[2], corner[3]),(55,255,155),2)
            cv2.imshow('img1', whole_img)

            res = sess.run(img_rois)
            # print(res[0]+98)
            cv2.imshow('img', (res[0]+98)/255)
            cv2.waitKey(0)
        # break
        # if i >= 10:
        if is_last_batch:
            break
        i += 1
    dataset.stop_loading()
    print('stop loading')
    produce_thread.join()
