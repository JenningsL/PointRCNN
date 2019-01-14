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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from kitti_object import *
from parameterize import obj_to_proposal_vec
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
            pc, mask, proposal_of_point = self.load_frame_data(frame_id)
            frame_data['frame_id'] = frame_id
            frame_data['pointcloud'] = pc
            frame_data['mask_label'] = mask
            frame_data['proposal_of_point'] = proposal_of_point
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

    def get_proposal_out(self, frame_data):
        objectness = np.zeros((self.npoints,), dtype=np.int32)
        center_cls = np.zeros((self.npoints,2), dtype=np.int32)
        angle_cls = np.zeros((self.npoints,), dtype=np.int32)
        size_cls = np.zeros((self.npoints,), dtype=np.int32)
        center_res = np.zeros((self.npoints, 3))
        angle_res = np.zeros((self.npoints,))
        size_res = np.zeros((self.npoints, 3))
        for i, prop in frame_data['proposal_of_point'].items():
            objectness[i] = 1
            center_cls[i] = prop[0]
            center_res[i] = prop[1]
            angle_cls[i] = prop[2]
            angle_res[i] = prop[3]
            size_cls[i] = prop[4]
            size_res[i] = prop[5]
        return objectness, center_cls, center_res, angle_cls, angle_res, size_cls, size_res

    def get_next_batch(self, bsize):
        is_last_batch = False
        total_batch = len(self.frame_ids) / bsize

        batch_data = np.zeros((bsize, self.npoints, self.num_channel))
        batch_label = np.zeros((bsize, self.npoints), dtype=np.int32)
        # proposal output for each point
        batch_objectness = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_center_x_cls = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_center_z_cls = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_center_x_res = np.zeros((bsize, self.npoints))
        batch_center_y_res = np.zeros((bsize, self.npoints))
        batch_center_z_res = np.zeros((bsize, self.npoints))
        batch_angle_cls = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_size_cls = np.zeros((bsize, self.npoints), dtype=np.int32)
        batch_angle_res = np.zeros((bsize, self.npoints))
        batch_size_res = np.zeros((bsize, self.npoints, 3))
        for i in range(bsize):
            frame = self.data_buffer.get()
            objectness, center_cls, center_res, angle_cls, angle_res, size_cls, size_res = \
                self.get_proposal_out(frame)
            batch_data[i,...] = frame['pointcloud']
            batch_label[i,:] = frame['mask_label']
            batch_objectness[i,...] = objectness
            batch_center_x_cls[i,...] = center_cls[:,0]
            batch_center_z_cls[i,...] = center_cls[:,1]
            batch_center_x_res[i,...] = center_res[:,0]
            batch_center_y_res[i,...] = center_res[:,1]
            batch_center_z_res[i,...] = center_res[:,2]
            batch_angle_cls[i,...] = angle_cls
            batch_size_cls[i,...] = size_cls
            # batch_center_res[i,...] = center_res
            batch_angle_res[i,...] = angle_res
            batch_size_res[i,...] = size_res
        if self.batch_idx == total_batch - 1:
            is_last_batch = True
            self.batch_idx = 0
        else:
            self.batch_idx += 1
        return batch_data, batch_label, batch_objectness, batch_center_x_cls,\
            batch_center_z_cls, batch_center_x_res, batch_center_y_res, \
            batch_center_z_res, batch_angle_cls, batch_angle_res, batch_size_cls, \
            batch_size_res, is_last_batch

    def load_frame_data(self, data_idx_str):
        '''load data for the first time'''
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
        # point index to proposal vector
        proposal_of_point = {}
        for obj in objects:
            _,obj_box_3d = utils.compute_box_3d(obj, calib.P)
            _,obj_mask = extract_pc_in_box3d(pc_rect, obj_box_3d)
            seg_mask[obj_mask] = 1
            obj_ids = np.where(obj_mask)[0]
            for idx in obj_ids:
                proposal_of_point[idx] = obj_to_proposal_vec(obj, pc_rect[idx,:3])

        return pc_rect, seg_mask, proposal_of_point

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
