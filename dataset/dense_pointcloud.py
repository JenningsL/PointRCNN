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
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'visualize/mayavi'))
from kitti_object import *
from data_conf import g_type2onehotclass, type_whitelist, difficulties_whitelist
from data_util import extract_pc_in_box3d
# from viz_util import draw_lidar


class DensePoints(object):
    """docstring for Stereo."""
    def __init__(self, kitti_path, img_size=(360, 1200)):
        super(DensePoints, self).__init__()
        self.kitti_path = kitti_path
        self.kitti_dataset = kitti_object(kitti_path, 'training')
        self.data_dir = os.path.join(self.kitti_path, 'training/velodyne_dense')
        self.img_size = img_size

    def to_point_cloud(self, depth_map, data_idx):
        calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        intrinsic = calib.P
        h = depth_map.shape[0]
        w = depth_map.shape[1]
        points = []
        border = 1
        for v in range(h):
            for u in range(w):
                if u < border or u >= w - border \
                    or v < border or v >= h - border:
                    continue
                z = depth_map[v][u]
                if z < 0:
                    continue
                x = (u - intrinsic[0][2])/intrinsic[0][0] * z
                y = (v - intrinsic[1][2])/intrinsic[1][1] * z
                points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def get_lidar(self, data_idx):
        # lidar point cloud
        pc_velo = self.kitti_dataset.get_lidar(data_idx)
        image = self.kitti_dataset.get_image(data_idx)
        pc_velo = self.kitti_dataset.get_lidar(data_idx)
        calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        img_height, img_width = image.shape[0:2]
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)
        pc_velo = pc_velo[img_fov_inds, :]
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        return pc_rect

    def load_dense_points(self, data_idx):
        fname = os.path.join(self.data_dir, '{:06}.npy'.format(data_idx))
        return np.load(fname)

    def generate_dense_points(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            frame_ids = [line.rstrip('\n') for line in f]
        depth_map_dir = os.path.join(self.kitti_path, 'training/depth_map_dense')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        for fid in frame_ids:
            fid = int(fid)
            depth_map = np.load(os.path.join(depth_map_dir, '{:06}.npy'.format(fid)))
            pc_dense = self.to_point_cloud(depth_map, fid)
            fout = os.path.join(self.data_dir, '{:06}.npy'.format(fid))
            np.save(fout, pc_dense)

            raw_pc = self.get_lidar(fid)
            print('{0}->{1}'.format(len(raw_pc), len(pc_dense)))
            # fig = draw_lidar(pc_dense)
            # draw_lidar(raw_pc, fig=fig, pts_color=(1, 1, 1))
            # raw_input()

    def generate_semantic_mask(self, split):
        output_dir = os.path.join(self.kitti_path, 'training/semantic_mask')
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            frame_ids = [line.rstrip('\n') for line in f]
        depth_map_dir = os.path.join(self.kitti_path, 'training/depth_map_dense')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for fid in frame_ids:
            fid = int(fid)
            depth_map = np.load(os.path.join(depth_map_dir, '{:06}.npy'.format(fid)))
            pc_dense = self.to_point_cloud(depth_map, fid)
            objects = self.kitti_dataset.get_label_objects(fid)
            image = self.kitti_dataset.get_image(fid)
            calib = self.kitti_dataset.get_calibration(fid) # 3 by 4 matrix
            objects = filter(lambda obj: obj.type in type_whitelist and obj.difficulty in difficulties_whitelist, objects)
            pts_mask = {
                'NonObject': np.zeros((len(pc_dense),), dtype=np.bool),
                'Car': np.zeros((len(pc_dense),), dtype=np.bool),
                'Pedestrian': np.zeros((len(pc_dense),), dtype=np.bool),
                'Cyclist': np.zeros((len(pc_dense),), dtype=np.bool)
            }
            for obj in objects:
                _,obj_box_3d = utils.compute_box_3d(obj, calib.P)
                _,mask = extract_pc_in_box3d(pc_dense, obj_box_3d)
                pts_mask[obj.type] = np.logical_or(mask, pts_mask[obj.type])
            pts_mask['NonObject'] = (pts_mask['Car'] + pts_mask['Pedestrian'] + pts_mask['Cyclist']) == 0

            semantic_mask = np.ones(image.shape[0:2], dtype=np.int32) * 255
            # disp_mask = np.ones(image.shape, dtype=np.int32) * 255
            # color_map = {
            #     'NonObject': (0,0,0),
            #     'Car': (0,0,142),
            #     'Pedestrian': (220, 20, 60),
            #     'Cyclist': (20,  255,  60)
            # }
            for k,v in pts_mask.items():
                pts_2d = calib.project_rect_to_image(pc_dense[v, :3])
                for u, v in pts_2d.astype(np.int32):
                    if u >= image.shape[1] or v >= image.shape[0]:
                        continue
                    semantic_mask[v, u] = g_type2onehotclass[k]
                    # disp_mask[v, u] = color_map[k]
            # cv2.imshow('semantic_mask', disp_mask/255.0)
            # cv2.waitKey(0)
            fout = os.path.join(output_dir, '{:06}.png'.format(fid))
            # np.save(fout, semantic_mask)
            cv2.imwrite(fout, semantic_mask)
            print('save', fid)

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    split = sys.argv[2]
    st = DensePoints(kitti_path)
    # st.generate_dense_points(split)
    st.generate_semantic_mask(split)
