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
                if u < border or u >= 1200 - border \
                    or v < border or v >= 360 - border:
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

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    split = sys.argv[2]
    st = DensePoints(kitti_path)
    st.generate_dense_points(split)
