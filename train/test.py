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
from rpn_dataset import Dataset
from train_util import compute_proposal_recall, compute_box3d_iou
from model_util import NUM_FG_POINT
from box_encoder import BoxEncoder
from rpn import RPN

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [default: 16384]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--split', default='val', help='Data split to use [default: val]')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
SPLIT = FLAGS.split

def log_string(out_str):
    print(out_str)


def test(split):
    is_training = False
    dataset = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', split, is_training=is_training)
    # data loading threads
    produce_thread = Thread(target=dataset.load, args=(False,))
    produce_thread.start()

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=())

            rpn_model = RPN(BATCH_SIZE, NUM_POINT, num_channel=4, is_training=is_training)
            pls = rpn_model.placeholders
            end_points = rpn_model.end_points

            box_center, box_angle, box_size = rpn_model.box_encoder.tf_decode(end_points)
            box_center = box_center + end_points['fg_points_xyz']
            #box_center = tf.reshape(box_center, [BATCH_SIZE * NUM_FG_POINT,3])
            #box_angle = tf.reshape(box_angle, [BATCH_SIZE * NUM_FG_POINT])
            #box_size = tf.reshape(box_size, [BATCH_SIZE * NUM_FG_POINT,3])

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
    fg_indices = []
    centers = []
    angles = []
    sizes = []
    proposal_boxes = []
    gt_boxes = []
    nms_indices = []
    scores = []
    segmentation = [] # point segmentation
    pc_choices = [] # point sampling indices

    while(True):
        batch_data, is_last_batch = dataset.get_next_batch(BATCH_SIZE, need_id=True)

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

        logits_val, indices_val, centers_val, angles_val, sizes_val, corners_val, ind_val, scores_val \
        = sess.run([
            end_points['foreground_logits'], end_points['fg_point_indices'],
            box_center, box_angle, box_size, end_points['proposal_boxes'],
            end_points['nms_indices'], end_points['proposal_scores']], feed_dict=feed_dict)
        print('inference time: ', datetime.now() - start)
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        num_batches += 1
        # results
        for i in range(BATCH_SIZE):
            frame_ids.append(batch_data['ids'][i])
            segmentation.append(preds_val[i])
            centers.append(centers_val[i])
            angles.append(angles_val[i])
            sizes.append(sizes_val[i])
            proposal_boxes.append(corners_val[i])
            nms_indices.append(ind_val[i])
            scores.append(scores_val[i])
            gt_boxes.append(batch_data['gt_boxes'][i])
            pc_choices.append(batch_data['pc_choice'][i])
        if is_last_batch:
        #if num_batches >= 500:
            break

    with open('rpn_out_{0}.pkl'.format(split),'wb') as fp:
        pickle.dump(frame_ids, fp)
        pickle.dump(segmentation, fp)
        pickle.dump(centers, fp)
        pickle.dump(angles, fp)
        pickle.dump(sizes, fp)
        pickle.dump(proposal_boxes, fp)
        pickle.dump(nms_indices, fp)
        pickle.dump(scores, fp)
        # pickle.dump(gt_boxes, fp)
        pickle.dump(pc_choices, fp)
    log_string('saved prediction')
    dataset.stop_loading()
    produce_thread.join()

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
    #TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val', types=['Car'], difficulties=[1])
    test(SPLIT)
