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
from kitti_object import *
from box_encoder import BoxEncoder
import kitti_util as utils
from data_util import rotate_points_along_y, shift_point_cloud, extract_pc_in_box3d
from data_conf import type_whitelist, difficulties_whitelist

NUM_HEADING_BIN = 12
NUM_CENTER_BIN = 12
CENTER_SEARCH_RANGE = 3.0
HEADING_SEARCH_RANGE = np.pi

box_encoder = BoxEncoder(CENTER_SEARCH_RANGE, NUM_CENTER_BIN, HEADING_SEARCH_RANGE, NUM_HEADING_BIN)

class Dataset(object):
    def __init__(self, npoints, kitti_path, split, \
        types=type_whitelist, difficulties=difficulties_whitelist):
        self.npoints = npoints
        self.kitti_path = kitti_path
        #self.batch_size = batch_size
        self.split = split
        self.kitti_dataset = kitti_object(kitti_path, 'training')
        self.frame_ids = self.load_split_ids(split)
        random.shuffle(self.frame_ids)
        # self.frame_ids = self.frame_ids[:100]
        self.num_channel = 4
        self.AUG_X = 1

        self.types_list = types
        self.difficulties_list = difficulties

        self.batch_idx = 0
        # preloading
        self.stop = False
        self.data_buffer = Queue(maxsize=64)

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def load(self, save_path, aug=False):
        i = 0
        while not self.stop:
            frame_id = self.frame_ids[i]
            #print('loading ' + frame_id)
            for x in range(self.AUG_X):
                frame_data = \
                    self.load_frame_data(frame_id, random_flip=aug, random_rotate=aug, random_shift=aug)
                frame_data['frame_id'] = frame_id
                self.data_buffer.put(frame_data)
            i = (i + 1) % len(self.frame_ids)

    def stop_loading(self):
        self.stop = True
        while not self.data_buffer.empty():
            item = self.data_buffer.get()
            self.data_buffer.task_done()

    def get_proposal_out(self, proposal_dict):
        '''assign the parameterized box to each point'''
        objectness = np.zeros((self.npoints,), dtype=np.int32)
        center_cls = np.zeros((self.npoints,2), dtype=np.int32)
        angle_cls = np.zeros((self.npoints,), dtype=np.int32)
        size_cls = np.zeros((self.npoints,), dtype=np.int32)
        center_res = np.zeros((self.npoints, 3))
        angle_res = np.zeros((self.npoints,))
        size_res = np.zeros((self.npoints, 3))
        for i, prop in proposal_dict.items():
            objectness[i] = 1
            center_cls[i] = prop[0]
            center_res[i] = prop[1]
            angle_cls[i] = prop[2]
            angle_res[i] = prop[3]
            size_cls[i] = prop[4]
            size_res[i] = prop[5]
        return objectness, center_cls, center_res, angle_cls, angle_res, size_cls, size_res

    def get_gt_box_of_points(self, box_dict):
        '''assign a ground truth box(corners) to each point'''
        boxes = np.zeros((self.npoints,8,3), dtype=np.float32)
        for i, box in box_dict.items():
            boxes[i] = box_dict[i]
        return boxes

    def get_next_batch(self, bsize, need_id=False):
        is_last_batch = False
        total_batch = len(self.frame_ids)*self.AUG_X / bsize

        batch = {
            'ids': [],
            'pointcloud': np.zeros((bsize, self.npoints, self.num_channel), dtype=np.float32),
            'images': np.zeros((bsize, 360, 1200, 3), dtype=np.float32),
            'calib': np.zeros((bsize, 3, 4), dtype=np.float32),
            'seg_label': np.zeros((bsize, self.npoints), dtype=np.int32),
            'prop_box': np.zeros((bsize, 7), dtype=np.float32),
            'center_x_cls': np.zeros((bsize,self.npoints), dtype=np.int32),
            'center_z_cls': np.zeros((bsize,self.npoints), dtype=np.int32),
            'center_x_res': np.zeros((bsize,self.npoints), dtype=np.float32),
            'center_y_res': np.zeros((bsize,self.npoints), dtype=np.float32),
            'center_z_res': np.zeros((bsize,self.npoints), dtype=np.float32),
            'angle_cls': np.zeros((bsize,self.npoints), dtype=np.int32),
            'size_cls': np.zeros((bsize,self.npoints), dtype=np.int32),
            'angle_res': np.zeros((bsize,self.npoints), dtype=np.float32),
            'size_res': np.zeros((bsize, self.npoints, 3), dtype=np.float32),
            'gt_box_of_point': np.zeros((bsize, self.npoints, 8, 3), dtype=np.float32),
            'gt_boxes': [],
            'pc_choice': [] # remember indices for sampling pointcloud in RCNN
        }
        for i in range(bsize):
            frame = self.data_buffer.get()
            batch['ids'].append(frame['frame_id'])
            objectness, center_cls, center_res, angle_cls, angle_res, size_cls, size_res = \
                frame['proposal_of_point']
            batch['pointcloud'][i,...] = frame['pointcloud']
            batch['images'][i,...] = frame['image']
            batch['calib'][i,...] = frame['calib']
            batch['seg_label'][i,:] = frame['mask_label']
            batch['center_x_cls'][i,...] = center_cls[:,0]
            batch['center_z_cls'][i,...] = center_cls[:,1]
            batch['center_x_res'][i,...] = center_res[:,0]
            batch['center_y_res'][i,...] = center_res[:,1]
            batch['center_z_res'][i,...] = center_res[:,2]
            batch['angle_cls'][i,...] = angle_cls
            batch['size_cls'][i,...] = size_cls
            # batch_center_res[i,...] = center_res
            batch['angle_res'][i,...] = angle_res
            batch['size_res'][i,...] = size_res
            batch['gt_box_of_point'][i,...] = frame['gt_box_of_point']
            batch['gt_boxes'].append(frame['gt_boxes'])
            batch['pc_choice'].append(frame['pc_choice'])
        if self.batch_idx == total_batch - 1:
            is_last_batch = True
            self.batch_idx = 0
            random.shuffle(self.frame_ids)
        else:
            self.batch_idx += 1
        return batch, is_last_batch

    def viz_frame(self, pc_rect, mask, gt_boxes):
        import mayavi.mlab as mlab
        from viz_util import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
        fig = draw_lidar(pc_rect)
        fig = draw_lidar(pc_rect[mask==1], fig=fig, pts_color=(1, 1, 1))
        fig = draw_gt_boxes3d(gt_boxes, fig, draw_text=False, color=(1, 1, 1))
        raw_input()

    def load_frame_data(self, data_idx_str,
        random_flip=False, random_rotate=False, random_shift=False):
        '''load one frame'''
        start = time.time()
        data_idx = int(data_idx_str)
        # print(data_idx_str)
        calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = self.kitti_dataset.get_label_objects(data_idx)
        image = self.kitti_dataset.get_image(data_idx)
        pc_velo = self.kitti_dataset.get_lidar(data_idx)
        img_height, img_width = image.shape[0:2]
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)
        pc_velo = pc_velo[img_fov_inds, :]
        #print(data_idx_str, pc_velo.shape[0])
        choice = np.random.choice(pc_velo.shape[0], self.npoints, replace=True)
        point_set = pc_velo[choice, :]
        pc_rect = np.zeros_like(point_set)
        pc_rect[:,0:3] = calib.project_velo_to_rect(point_set[:,0:3])
        pc_rect[:,3] = point_set[:,3]
        seg_mask = np.zeros((pc_rect.shape[0]))
        objects = filter(lambda obj: obj.type in self.types_list and obj.difficulty in self.difficulties_list, objects)
        gt_boxes = [] # ground truth boxes
        # data augmentation
        if random_flip and np.random.random()>0.5: # 50% chance flipping
            pc_rect[:,0] *= -1
            for obj in objects:
                obj.t = [-obj.t[0], obj.t[1], obj.t[2]]
                # ensure that ry is [-pi, pi]
                if obj.ry >= 0:
                    obj.ry = np.pi - obj.ry
                else:
                    obj.ry = -np.pi - obj.ry

        if random_rotate:
            ry = (np.random.random() - 0.5) * math.radians(20) # -10~10 degrees
            pc_rect[:,0:3] = rotate_points_along_y(pc_rect[:,0:3], ry)
            for obj in objects:
                obj.t = rotate_points_along_y(obj.t, ry)
                obj.ry -= ry
                # ensure that ry is [-pi, pi]
                if obj.ry > np.pi:
                    obj.ry -= 2*np.pi
                elif obj.ry < -np.pi:
                    obj.ry += 2*np.pi
        proposal_of_point = {} # point index to proposal vector
        gt_box_of_point = {} # point index to corners_3d
        for obj in objects:
            _,obj_box_3d = utils.compute_box_3d(obj, calib.P)
            _,obj_mask = extract_pc_in_box3d(pc_rect, obj_box_3d)
            if np.sum(obj_mask) == 0:
                # label without 3d points
                # print('skip object without points')
                continue
            seg_mask[obj_mask] = 1
            gt_boxes.append(obj_box_3d)
            obj_idxs = np.where(obj_mask)[0]
            # data augmentation
            # FIXME: jitter point will make valid loss growing
            if random_shift and False: # jitter object points
                pc_rect[obj_idxs,:3] = shift_point_cloud(pc_rect[obj_idxs,:3], 0.02)
            for idx in obj_idxs:
                proposal_of_point[idx] = box_encoder.encode(obj, pc_rect[idx,:3])
                gt_box_of_point[idx] = obj_box_3d
        # self.viz_frame(pc_rect, seg_mask, gt_boxes)
        # return pc_rect, seg_mask, proposal_of_point, gt_box_of_point, gt_boxes
        calib_matrix = np.copy(calib.P)
        calib_matrix[0,:] *= (1200.0 / image.shape[1])
        calib_matrix[1,:] *= (360.0 / image.shape[0])
        return {
            'pointcloud': pc_rect,
            'image': cv2.resize(image, (1200, 360)),
            'calib': calib_matrix,
            'mask_label': seg_mask,
            'proposal_of_point': self.get_proposal_out(proposal_of_point),
            'gt_box_of_point': self.get_gt_box_of_points(gt_box_of_point),
            'gt_boxes': gt_boxes,
            'pc_choice': choice
        }

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    split = sys.argv[2]

    sys.path.append('../models')
    from collections import namedtuple
    import tensorflow as tf
    from img_vgg_pyramid import ImgVggPyr
    import projection
    VGG_config = namedtuple('VGG_config', 'vgg_conv1 vgg_conv2 vgg_conv3 vgg_conv4 l2_weight_decay')

    # dataset = Dataset(16384, kitti_path, split, types=['Car'], difficulties=[1])
    dataset = Dataset(16384, kitti_path, split)
    # dataset.load('./train', True)
    produce_thread = threading.Thread(target=dataset.load, args=('./train',False))
    produce_thread.start()
    i = 0
    total = 0
    while(True):
        batch_data, is_last_batch = dataset.get_next_batch(1, need_id=True)
        with tf.Session() as sess:
            img_vgg = ImgVggPyr(VGG_config(**{
                'vgg_conv1': [2, 32],
                'vgg_conv2': [2, 64],
                'vgg_conv3': [3, 128],
                'vgg_conv4': [3, 256],
                'l2_weight_decay': 0.0005
            }))

            pts2d = projection.tf_rect_to_image(tf.slice(batch_data['pointcloud'],[0,0,0],[-1,-1,3]), batch_data['calib'])
            pts2d = tf.cast(pts2d, tf.int32) #(B,N,2)
            indices = tf.concat([
                tf.expand_dims(tf.tile(tf.range(0, 1), [16384]), axis=-1), # (B*N, 1)
                tf.reshape(pts2d, [1*16384, 2])
            ], axis=-1) # (B*N,3)
            indices = tf.gather(indices, [0,2,1], axis=-1) # image's shape is (y,x)
            point_img_feats = tf.reshape(
                tf.gather_nd(batch_data['images'], indices), # (B*N,C)
                [1, 16384, -1])  # (B,N,C)
            res = sess.run(point_img_feats)
            p2d = sess.run(pts2d)
            print(batch_data['images'][0,p2d[0][0][1],p2d[0][0][0]])
            print(res[0][0])
            break
            img = batch_data['images'][0]/255
            for i,p in enumerate(res[0]):
                if batch_data['seg_label'][0][i] != 1:
                    continue
                cv2.circle(img,(p[0], p[1]),1,(255,0,0),1)
            cv2.imshow('img', img)
            cv2.waitKey(0)

        if is_last_batch:
            break
        i += 1
    dataset.stop_loading()
    print('stop loading')
    produce_thread.join()
