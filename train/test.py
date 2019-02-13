from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pickle
from threading import Thread
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from kitti import Dataset
from train_util import compute_proposal_recall, compute_box3d_iou
from rpn import RPN

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [default: 16384]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

def log_string(out_str):
    print(out_str)

TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val', types=['Car'], difficulties=[1])

def test():
    # data loading threads
    test_produce_thread = Thread(target=TEST_DATASET.load, args=('/data/ssd/public/jlliu/PointRCNN/dataset/val',False))
    test_produce_thread.start()

    is_training = False
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=())

            rpn_model = RPN(BATCH_SIZE, NUM_POINT, num_channel=4, is_training=is_training)
            pls = rpn_model.placeholders
            end_points = rpn_model.end_points
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, FLAGS.model_path)

    log_string(str(datetime.now()))

    # To collect statistics
    total_correct = 0
    total_seen = 0
    num_batches = 0

    frame_ids = []
    pointclouds = []
    preds = []
    labels = []
    fg_points = []
    fg_indices = []
    proposal_boxes = []
    gt_boxes = []
    point_gt_boxes = []
    nms_indices = []

    while(True):
        batch_data, is_last_batch = TEST_DATASET.get_next_batch(BATCH_SIZE, need_id=True)

        feed_dict = {
            pls['pointclouds']: batch_data['pointcloud'],
            pls['img_inputs']: batch_data['images'],
            pls['calib']: batch_data['calib'],
            pls['seg_labels']: batch_data['seg_label'],
            pls['center_bin_x_labels']: batch_data['center_x_cls'],
            pls['center_bin_z_labels']: batch_data['center_z_cls'],
            pls['center_x_residuals_labels']: batch_data['center_x_res'],
            pls['center_y_residuals_labels']: batch_data['center_y_res'],
            pls['center_z_residuals_labels']: batch_data['center_z_res'],
            pls['heading_bin_labels']: batch_data['angle_cls'],
            pls['heading_residuals_labels']: batch_data['angle_res'],
            pls['size_class_labels']: batch_data['size_cls'],
            pls['size_residuals_labels']: batch_data['size_res'],
            pls['gt_box_of_point']: batch_data['gt_box_of_point'],
            pls['is_training_pl']: is_training,
        }

        start = datetime.now()
        logits_val, points_val, indices_val, boxes_val, pts_boxes_val, ind_val = sess.run([
            end_points['foreground_logits'], end_points['fg_points'],
            end_points['fg_point_indices'], end_points['proposal_boxes'],
            end_points['gt_box_of_point'], end_points['nms_indices']], feed_dict=feed_dict)
        print('inference time: ', datetime.now() - start)
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        num_batches += 1
        # results
        for i in range(BATCH_SIZE):
            frame_ids.append(batch_data['ids'][i])
            #pointclouds.append(batch_pc[i])
            #preds.append(preds_val[i])
            labels.append(batch_data['seg_label'][i])
            #fg_points.append(points_val[i])
            #fg_indices.append(indices_val[i])
            proposal_boxes.append(boxes_val[i])
            gt_boxes.append(batch_data['gt_boxes'][i])
            #point_gt_boxes.append(pts_boxes_val[i])
            nms_indices.append(ind_val[i])
        if is_last_batch:
        #if num_batches >= 500:
            break

    with open('prediction.pkl','wb') as fp:
        pickle.dump(frame_ids, fp)
        pickle.dump(proposal_boxes, fp)
        pickle.dump(gt_boxes, fp)
        pickle.dump(nms_indices, fp)
        #pickle.dump(point_gt_boxes, fp)
    log_string('saved prediction')
    TEST_DATASET.stop_loading()
    test_produce_thread.join()

    '''
    all_indices = np.tile(np.arange(1024), (len(proposal_boxes),))
    iou2d, iou3d = compute_box3d_iou(proposal_boxes, point_gt_boxes, all_indices)
    print('IOU2d: ', np.mean(iou2d))
    print('IOU3d: ', np.mean(iou3d))
    '''
    recall = compute_proposal_recall(proposal_boxes, gt_boxes, nms_indices)
    print('Average recall: ', recall)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
