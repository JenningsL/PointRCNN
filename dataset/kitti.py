from __future__ import print_function

import os
import sys
import numpy as np
import copy
import random
import threading
from Queue import Queue
import time
import cPickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
from kitti_object import *
import kitti_util as utils
type_whitelist = ['Car', 'Pedestrian', 'Cyclist']

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

class Dataset(object):
    def __init__(self, npoints, kitti_path, split):
        self.npoints = npoints
        self.kitti_path = kitti_path
        #self.batch_size = batch_size
        self.split = split
        self.kitti_dataset = kitti_object(kitti_path, 'training')
        self.frame_ids = self.load_split_ids(split)
        #self.frame_ids = self.frame_ids[:20]
        self.num_channel = 4

        self.batch_idx = 0
        # preloading
        self.stop = False
        self.data_buffer = Queue(maxsize=128)

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def preprocess(self, save_path):
        frame_data = {}
        for frame_id in self.frame_ids:
            pc, mask = self.load_frame_data(frame_id)
            frame_data['frame_id'] = frame_id
            frame_data['pointcloud'] = pc
            frame_data['mask_label'] = mask
            with open(os.path.join(save_path, frame_id+'.pkl'),'wb') as fp:
                pickle.dump(frame_data, fp)

    def load(self, save_path):
        i = 0
        while not self.stop:
            frame_id = self.frame_ids[i]
            #print('loading ' + frame_id)
            with open(os.path.join(save_path, frame_id+'.pkl'), 'rb') as f:
                frame_data = pickle.load(f)
                self.data_buffer.put(frame_data)
            i = (i + 1) % len(self.frame_ids)

    def stop_loading(self):
        self.stop = True
        while not self.data_buffer.empty():
            item = self.data_buffer.get()
            self.data_buffer.task_done()

    def get_next_batch(self, bsize):
        is_last_batch = False
        total_batch = len(self.frame_ids) / bsize

        batch_data = np.zeros((bsize, self.npoints, self.num_channel))
        batch_label = np.zeros((bsize, self.npoints), dtype=np.int32)
        for i in range(bsize):
            frame = self.data_buffer.get()
            batch_data[i,...] = frame['pointcloud']
            batch_label[i,:] = frame['mask_label']
        if self.batch_idx == total_batch - 1:
            is_last_batch = True
            self.batch_idx = 0
        else:
            self.batch_idx += 1
        return batch_data, batch_label, is_last_batch

    def load_frame_data(self, data_idx_str):
        '''load data for the first time'''
        # if os.path.exists(os.path.join(self.save_dir, frame_id+'.pkl')):
        #     with open(os.path.join(self.save_dir, frame_id+'.pkl'), 'rb') as f:
        #         return pickle.load(f)
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
        objects = filter(lambda obj: obj.type in type_whitelist, objects)
        for obj in objects:
            _,obj_box_3d = utils.compute_box_3d(obj, calib.P)
            _,obj_inds = extract_pc_in_box3d(pc_rect, obj_box_3d)
            seg_mask[obj_inds] = 1
        return pc_rect, seg_mask

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    split = sys.argv[2]
    dataset = Dataset(20000, kitti_path, split)
    dataset.preprocess(split)

    '''
    produce_thread = threading.Thread(target=dataset.load, args=('./train',))
    produce_thread.start()
    while(True):
        batch_data, batch_label, is_last = dataset.get_next_batch(1)
        print(batch_data.shape, batch_label.shape, is_last)
        if is_last:
            break
    dataset.stop_loading()
    print('stop loading')
    produce_thread.join()
    '''
