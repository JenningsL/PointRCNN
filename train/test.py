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
from train_util import compute_proposal_recall

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [default: 16384]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

def log_string(out_str):
    print(out_str)

TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val', types=['Car'], difficulties=[1])

def test():
    # data loading threads
    test_produce_thread = Thread(target=TEST_DATASET.load, args=('/data/ssd/public/jlliu/PointRCNN/dataset/val',False))
    test_produce_thread.start()

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, mask_labels_pl, \
            center_bin_x_pl, center_bin_z_pl,\
            center_x_residuals_pl, center_z_residuals_pl, center_y_residuals_pl, heading_bin_pl,\
            heading_residuals_pl, size_class_pl, size_residuals_pl, \
            gt_boxes_pl, gt_box_of_point_pl \
                = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Get model
            end_points = {}
            labels_pl = {
                'mask_label': mask_labels_pl,
                'center_bin_x': center_bin_x_pl,
                'center_bin_z': center_bin_z_pl,
                'center_x_residuals': center_x_residuals_pl,
                'center_z_residuals': center_z_residuals_pl,
                'center_y_residuals': center_y_residuals_pl,
                'heading_bin': heading_bin_pl,
                'heading_residuals': heading_residuals_pl,
                'size_class': size_class_pl,
                'gt_box_of_point': gt_box_of_point_pl,
                'size_residuals': size_residuals_pl
            }
            end_points = MODEL.get_model(pointclouds_pl, labels_pl,
                is_training_pl, None, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, FLAGS.model_path)

        ops = {
            'pointclouds_pl': pointclouds_pl,
            'is_training_pl': is_training_pl,
            'step': batch,
            'end_points': end_points}
        ops.update(labels_pl)

    is_training = False
    #is_training = True
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
    nms_indices = []

    while(True):
        batch_pc, batch_mask_label, \
        batch_center_bin_x, batch_center_bin_z, batch_center_x_residuals, \
        batch_center_y_residuals, batch_center_z_residuals, batch_heading_bin, \
        batch_heading_residuals, batch_size_class, batch_size_residuals, batch_gt_boxes, \
        batch_gt_box_of_point, batch_ids, is_last_batch \
        = TEST_DATASET.get_next_batch(BATCH_SIZE, need_id=True)

        feed_dict = {
            ops['pointclouds_pl']: batch_pc,
            ops['mask_label']: batch_mask_label,
            ops['center_bin_x']: batch_center_bin_x,
            ops['center_bin_z']: batch_center_bin_z,
            ops['center_x_residuals']: batch_center_x_residuals,
            ops['center_y_residuals']: batch_center_y_residuals,
            ops['center_z_residuals']: batch_center_z_residuals,
            ops['heading_bin']: batch_heading_bin,
            ops['heading_residuals']: batch_heading_residuals,
            ops['size_class']: batch_size_class,
            ops['size_residuals']: batch_size_residuals,
            ops['gt_box_of_point']: batch_gt_box_of_point,
            ops['is_training_pl']: is_training,
        }

        start = datetime.now()
        logits_val, points_val, indices_val, boxes_val, ind_val = sess.run([
            ops['end_points']['foreground_logits'], ops['end_points']['fg_points'],
            ops['end_points']['fg_point_indices'], ops['end_points']['proposal_boxes'],
            ops['end_points']['nms_indices']], feed_dict=feed_dict)
        print('inference time: ', datetime.now() - start)
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        num_batches += 1
        # results
        for i in range(BATCH_SIZE):
            frame_ids.append(batch_ids[i])
            pointclouds.append(batch_pc[i])
            preds.append(preds_val[i])
            labels.append(batch_mask_label[i])
            fg_points.append(points_val[i])
            fg_indices.append(indices_val[i])
            proposal_boxes.append(boxes_val[i])
            gt_boxes.append(batch_gt_boxes[i])
            nms_indices.append(ind_val[i])
        if is_last_batch:
        #if num_batches >= 500:
            break

    with open('prediction.pkl','wb') as fp:
        pickle.dump(frame_ids, fp)
        pickle.dump(proposal_boxes, fp)
        pickle.dump(gt_boxes, fp)
        pickle.dump(nms_indices, fp)
    log_string('saved prediction')
    TEST_DATASET.stop_loading()
    test_produce_thread.join()

    recall = compute_proposal_recall(proposal_boxes, gt_boxes, nms_indices)
    print('Average recall: ', recall)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()
