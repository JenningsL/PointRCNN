from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import pickle
from threading import Thread
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
from rcnn_dataset import Dataset
import train_util
from rcnn import RCNN
import kitti_util as utils
from shapely.geometry import Polygon

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=512, help='Point Number [default: 512]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--kitti_path', default='/data/ssd/public/jlliu/Kitti/object', help='Kitti root path')
parser.add_argument('--split', default='val', help='Data split to use [default: val]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
KITTI_PATH = FLAGS.kitti_path
SPLIT = FLAGS.split

def log_string(out_str):
    print(out_str)

#TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val', is_training=False)
TEST_DATASET = Dataset(NUM_POINT, KITTI_PATH, SPLIT, is_training=(SPLIT in ['train', 'val']))
type_list = ['NonObject', 'Car', 'Pedestrian', 'Cyclist']

calib_cache = {}
def get_calibration(idx):
    if idx not in calib_cache:
        calib_cache[idx] = TEST_DATASET.kitti_dataset.get_calibration(idx)
    return calib_cache[idx]

class DetectObject(object):
    def __init__(self, h,w,l,tx,ty,tz,ry, frame_id, type_label, score, box_2d=None, box_3d=None):
        self.t = [tx,ty,tz]
        self.ry = ry
        self.h = h
        self.w = w
        self.l = l
        self.frame_id = frame_id
        self.type_label = type_label
        self.score = score
        self.box_2d = box_2d
        self.box_3d = box_3d # corners

def test():
    ''' Main function for training and simple evaluation. '''
    result_dir = FLAGS.output
    # data loading threads
    test_produce_thread = Thread(target=TEST_DATASET.load, args=(False,))
    test_produce_thread.start()

    is_training = False
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            rcnn_model = RCNN(BATCH_SIZE, NUM_POINT, TEST_DATASET.num_channel, is_training=is_training)
            pls = rcnn_model.placeholders

            # Get model and losses
            end_points = rcnn_model.end_points
            loss, loss_endpoints = rcnn_model.get_loss()

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, FLAGS.model_path)

    objects = {}
    boxes = []
    while(True):
        batch_data, is_last_batch = TEST_DATASET.get_next_batch(BATCH_SIZE)
        feed_dict = {
            pls['pointclouds']: batch_data['pointcloud'],
            pls['img_inputs']: batch_data['images'],
            pls['img_seg_map']: batch_data['img_seg_map'],
            pls['calib']: batch_data['calib'],
            pls['proposal_boxes']: batch_data['prop_box'],
            pls['class_labels']: batch_data['label'],
            pls['center_bin_x_labels']: batch_data['center_x_cls'],
            pls['center_bin_z_labels']: batch_data['center_z_cls'],
            pls['center_x_res_labels']: batch_data['center_x_res'],
            pls['center_y_res_labels']: batch_data['center_y_res'],
            pls['center_z_res_labels']: batch_data['center_z_res'],
            pls['heading_bin_labels']: batch_data['angle_cls'],
            pls['heading_res_labels']: batch_data['angle_res'],
            pls['size_class_labels']: batch_data['size_cls'],
            pls['size_res_labels']: batch_data['size_res'],
            pls['gt_box_of_prop']: batch_data['gt_box_of_prop'],
            pls['is_training_pl']: is_training
        }

        cls_logits, box_center, box_angle, box_size, box_score, corners = \
                sess.run([end_points['cls_logits'],
                end_points['box_center'], end_points['box_angle'], end_points['box_size'],
                end_points['box_score'], end_points['box_corners']], feed_dict=feed_dict)
        cls_val = np.argmax(cls_logits, axis=-1)
        correct = np.sum(cls_val == batch_data['label'])
        for i in range(BATCH_SIZE):
            if type_list[cls_val[i]] == 'NonObject':
                #print('NonObject')
                continue
            idx = int(batch_data['ids'][i])
            size = box_size[i]
            angle = box_angle[i]
            center = box_center[i]
            box_corner = corners[i]
            score = box_score[i]
            obj = DetectObject(size[1],size[2],size[0],center[0],center[1],center[2],angle,idx,type_list[cls_val[i]],score)
            # dont't use batch_data['calib'] which is for resized image
            calib = get_calibration(idx).P
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib)
            if box3d_pts_2d is None:
                print('box3d_pts_2d is None')
                continue
            x1 = np.amin(box3d_pts_2d, axis=0)[0]
            y1 = np.amin(box3d_pts_2d, axis=0)[1]
            x2 = np.amax(box3d_pts_2d, axis=0)[0]
            y2 = np.amax(box3d_pts_2d, axis=0)[1]
            obj.box_2d = [x1,y1,x2,y2]
            obj.box_3d = box3d_pts_3d
            if idx not in objects:
                objects[idx] = []
            objects[idx].append(obj)

            boxes.append(corners[i])
        if is_last_batch:
            break

    TEST_DATASET.stop_loading()
    test_produce_thread.join()

    with open('rcnn_out.pkl','wb') as fp:
        pickle.dump(objects, fp)
    objects = nms_on_bev(objects, 0.01)
    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, objects)
    output_dir = os.path.join(result_dir, 'data')
    print('write detection results to ' + output_dir)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    to_fill_filename_list = ['%06d.txt'%(int(frame_id)) \
            for frame_id in TEST_DATASET.frame_ids]
    fill_files(output_dir, to_fill_filename_list)

def nms_on_bev(objects, iou_threshold=0.1):
    final_result = {}
    for frame_id, detections in objects.items():
        final_result[frame_id] = []
        calib = get_calibration(int(frame_id))
        #detections = filter(lambda d: d.score > np.log(0.5**4) and d.probs[2] >= 0.3, detections)
        groups = group_overlaps(detections, calib, iou_threshold)
        for group in groups:
            # highest score
            group = sorted(group, key=lambda o: o.score, reverse=True)
            keep = group[0]
            final_result[frame_id].append(keep)
    return final_result

def get_iou(bev_box1, bev_box2):
    p1 = Polygon(bev_box1)
    p2 = Polygon(bev_box2)
    intersection = p1.intersection(p2).area
    return intersection / (p1.area + p2.area - intersection)

def group_overlaps(detections, calib, iou_thres=0.01):
    bev_boxes = map(lambda obj: utils.compute_box_3d(obj, calib.P)[1][:4, [0,2]], detections)
    groups = []
    candidates = range(len(detections))
    while len(candidates) > 0:
        idx = candidates[0]
        group = [idx]
        for i in candidates[1:]:
            if get_iou(bev_boxes[idx], bev_boxes[i]) >= iou_thres:
                group.append(i)
        for j in group:
            candidates.remove(j)
        groups.append(map(lambda i: detections[i], group))
    return groups

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def write_detection_results(result_dir, detection_objects):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for idx, obj_list in detection_objects.items():
        results[idx] = []
        for obj in obj_list:
            output_str = obj.type_label + " -1 -1 -10 "
            box2d = obj.box_2d
            output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
            output_str += "%f %f %f %f %f %f %f %f" % (obj.h,obj.w,obj.l,obj.t[0],obj.t[1],obj.t[2],obj.ry,obj.score)
            results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close()

def load():
    result_dir = FLAGS.output
    with open('rcnn_out.pkl','rb') as fp:
        objects = pickle.load(fp)
    for k,v in objects.items():
        if len(v) > 0:
            print(k)
    objects = nms_on_bev(objects, 0.01)
    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, objects)
    output_dir = os.path.join(result_dir, 'data')
    print('write detection results to ' + output_dir)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    to_fill_filename_list = ['%06d.txt'%(int(frame_id)) \
            for frame_id in TEST_DATASET.frame_ids]
    fill_files(output_dir, to_fill_filename_list)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
    #load()
