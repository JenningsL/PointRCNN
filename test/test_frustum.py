''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import cPickle as pickle
from threading import Thread
from shapely.geometry import Polygon
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
from frustum_model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from frustum_model_util import NUM_SEG_CLASSES, NUM_OBJ_CLASSES, g_type2onehotclass, type_whitelist
from frustum_dataset import FrustumDataset, Sample
import provider
from kitti_object import *
import kitti_util as utils
from frustum_pointnets_v2 import FrustumPointNet


def get_session_and_model(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            frustum_pointnet = FrustumPointNet(batch_size, num_point)
            placeholders = frustum_pointnet.placeholders
            end_points = frustum_pointnet.end_points
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
    return sess, frustum_pointnet

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, model, pc, img_seg_map, prop_box, calib, cls_label):
    ''' Run inference for frustum pointnets in batch mode '''
    #assert pc.shape[0] == BATCH_SIZE

    ep = model.end_points
    pls = model.placeholders

    feed_dict = {pls['pointclouds']: pc,
                 pls['img_seg_map']: img_seg_map,
                 pls['prop_box']: prop_box,
                 pls['calib']: calib,
                 pls['cls_label']: cls_label,
                 pls['is_training']: False}
    cls_logits, logits, centers, \
    heading_logits, heading_residuals, \
    size_logits, size_residuals = \
        sess.run([ep['cls_logits'], ep['mask_logits'], ep['center'],
            ep['heading_scores'], ep['heading_residuals'],
            ep['size_scores'], ep['size_residuals']],
            feed_dict=feed_dict)

    # Compute scores
    smooth = 0.000001
    batch_cls_prob = np.max(softmax(cls_logits),1) # B,
    batch_seg_prob = softmax(logits)[:,:,1] # BxN
    batch_seg_mask = np.argmax(logits, 2) # BxN
    mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
    mask_mean_prob = mask_mean_prob / (np.sum(batch_seg_mask,1) + smooth) # B,
    heading_prob = np.max(softmax(heading_logits),1) # B
    size_prob = np.max(softmax(size_logits),1) # B,
    scores = np.log(batch_cls_prob) + np.log(mask_mean_prob + smooth) + np.log(heading_prob) + np.log(size_prob)
    #scores = (batch_cls_prob + mask_mean_prob + heading_prob + size_prob) / 4
    probs = np.column_stack((batch_cls_prob, mask_mean_prob, heading_prob, size_prob))
    # Finished computing scores

    type_cls = np.argmax(cls_logits, 1)
    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return type_cls, centers, heading_cls, heading_res, \
        size_cls, size_res, scores, probs

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

def to_detection_objects(id_list, type_list, center_list, \
                        heading_cls_list, heading_res_list, \
                        size_cls_list, size_res_list, \
                        rot_angle_list, score_list, prob_list, proposal_score_list, dataset):
    objects = {}
    for i in range(len(center_list)):
        if type_list[i] == 'NonObject':
            continue
        idx = id_list[i]
        #score = score_list[i] + np.log(proposal_score_list[i])
        score = score_list[i]
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        obj = DetectObject(h,w,l,tx,ty,tz,ry,idx,type_list[i],score)
        # cal 2d box from 3d box
        calib = dataset.kitti_dataset.get_calibration(idx)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            continue
        x1 = np.amin(box3d_pts_2d, axis=0)[0]
        y1 = np.amin(box3d_pts_2d, axis=0)[1]
        x2 = np.amax(box3d_pts_2d, axis=0)[0]
        y2 = np.amax(box3d_pts_2d, axis=0)[1]
        obj.box_2d = [x1,y1,x2,y2]
        obj.box_3d = box3d_pts_3d
        obj.probs = prob_list[i]
        obj.proposal_score = proposal_score_list[i]

        if idx not in objects:
            objects[idx] = []
        objects[idx].append(obj)
    return objects

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

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

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

def nms_on_bev(objects, dataset, iou_threshold=0.1):
    final_result = {}
    for frame_id, detections in objects.items():
        final_result[frame_id] = []
        calib = dataset.kitti_dataset.get_calibration(int(frame_id))
        #detections = filter(lambda d: d.score > np.log(0.5**4) and d.probs[2] >= 0.3, detections)
        groups = group_overlaps(detections, calib, iou_threshold)
        for group in groups:
            # highest score
            group = sorted(group, key=lambda o: o.score, reverse=True)
            keep = group[0]
            final_result[frame_id].append(keep)
    return final_result

def test(TEST_DATASET, output_filename, result_dir=None):
    ''' Test frustum pointents with 2D boxes from a RGB detector.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    val_loading_thread = Thread(target=TEST_DATASET.load)
    val_loading_thread.start()

    batch_size = TEST_DATASET.batch_size
    num_point = TEST_DATASET.npoints

    #ps_list = []
    cls_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    prob_list = []
    frame_id_list = []
    proposal_score_list = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    sess, model = get_session_and_model(batch_size, num_point)
    # for batch_idx in range(num_batches):
    batch_idx = 0
    while(True):
        batch_data, is_last_batch = TEST_DATASET.get_next_batch()
        print('batch idx: %d' % (batch_idx))
        batch_idx += 1

        # FIXME: will discard last batch if it has less samples than batch size
        if len(batch_data['ids']) != batch_size:
            break

        # Run one batch inference
    	batch_cls, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores, batch_prob = \
            inference(sess, model, batch_data['pointcloud'], batch_data['img_seg_map'],
            batch_data['prop_box'], batch_data['calib'], batch_data['cls_label'])

        tp = np.sum(np.logical_and(batch_cls == batch_data['cls_label'], batch_data['cls_label'] < g_type2onehotclass['NonObject']))
        fp = np.sum(np.logical_and(batch_cls != batch_data['cls_label'], batch_data['cls_label'] == g_type2onehotclass['NonObject']))
        fn = np.sum(np.logical_and(batch_cls != batch_data['cls_label'], batch_data['cls_label'] < g_type2onehotclass['NonObject']))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        if total_tp+total_fn > 0 and total_tp+total_fp > 0:
            print('average recall: {}, precision: {}'.format(float(total_tp)/(total_tp+total_fn), float(total_tp)/(total_tp+total_fp)))

        for i in range(batch_size):
            #ps_list.append(batch_data[i,...])
            cls_list.append(batch_cls[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_data['rot_angle'][i])
            score_list.append(batch_scores[i])
            prob_list.append(batch_prob[i])
            proposal_score_list.append(batch_data['proposal_score'][i])
        frame_id_list += map(lambda fid: int(fid), batch_data['ids'])
        if is_last_batch:
            break

    type_list = map(lambda i: type_whitelist[i], cls_list)

    detection_objects = to_detection_objects(frame_id_list, type_list,
        center_list, heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list, prob_list,
        proposal_score_list, TEST_DATASET)
    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(detection_objects, fp)
    detection_objects = nms_on_bev(detection_objects, TEST_DATASET, 0.1)
    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, detection_objects)
    output_dir = os.path.join(result_dir, 'data')
    print('write detection results to ' + output_dir)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    to_fill_filename_list = [frame_id+'.txt' \
            for frame_id in TEST_DATASET.frame_ids]
    fill_files(output_dir, to_fill_filename_list)
    TEST_DATASET.stop_loading()
    val_loading_thread.join()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
    parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
    parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
    parser.add_argument('--kitti_path', default='/data/ssd/public/jlliu/Kitti/object', help='Kitti root path')
    parser.add_argument('--split', default='val', help='Data split to use [default: val]')
    # parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
    parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
    parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
    FLAGS = parser.parse_args()

    # Set training configurations
    BATCH_SIZE = FLAGS.batch_size
    MODEL_PATH = FLAGS.model_path
    GPU_INDEX = FLAGS.gpu
    NUM_POINT = FLAGS.num_point

    TEST_DATASET = FrustumDataset(NUM_POINT, FLAGS.kitti_path, BATCH_SIZE, FLAGS.split,
                 data_dir='./rcnn_data_'+FLAGS.split, is_training=False,
                 augmentX=1, random_shift=False, rotate_to_center=True, random_flip=False, use_gt_prop=False)

    test(TEST_DATASET, FLAGS.output+'.pickle', FLAGS.output)
