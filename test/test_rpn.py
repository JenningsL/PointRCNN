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
sys.path.append(os.path.join(ROOT_DIR, 'train'))
from data_conf import g_type2onehotclass
from rpn_dataset import Dataset
from train_util import compute_proposal_recall, compute_box3d_iou
from model_util import NUM_FG_POINT
from box_encoder import BoxEncoder
from rpn import RPN, NUM_SEG_CLASSES
from img_seg_net_deeplab import ImgSegNet

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [default: 16384]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--img_model_path', default=None, help='Restore image seg model path e.g. ./frozen_inference_graph.pb [default: None]')
parser.add_argument('--kitti_path', default='/data/ssd/public/jlliu/Kitti/object', help='Kitti root path')
parser.add_argument('--split', default='val', help='Data split to use [default: val]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
KITTI_PATH = FLAGS.kitti_path
SPLIT = FLAGS.split

if FLAGS.no_intensity:
    NUM_CHANNEL = 3
else:
    NUM_CHANNEL = 4

def log_string(out_str):
    print(out_str)


def test(split, save_result=False):
    if save_result and not os.path.exists('./rcnn_data_'+split):
        os.mkdir('./rcnn_data_'+split)
    is_training = False
    #dataset = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', split, is_training=is_training)
    dataset = Dataset(NUM_POINT, NUM_CHANNEL, KITTI_PATH, split, is_training=(split in ['train', 'val']), use_aug_scene=False)
    # data loading threads
    produce_thread = Thread(target=dataset.load, args=(False,))
    produce_thread.start()

    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
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

    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            #img_seg_net = ImgSegNet(BATCH_SIZE, NUM_POINT, num_channel=4, bn_decay=None, is_training=is_training)
            #seg_softmax = img_seg_net.get_seg_softmax()
            #saver1 = tf.train.Saver()
            img_seg_net = ImgSegNet(BATCH_SIZE, NUM_POINT)
            #img_seg_net.load_graph('/data/ssd/public/jlliu/models/research/deeplab/exp/frozen_inference_graph.pb')
            img_seg_net.load_graph(FLAGS.img_model_path)
            seg_softmax = img_seg_net.get_seg_softmax()
            full_seg = img_seg_net.get_semantic_seg()
        # Create another session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess1 = tf.Session(config=config)

        #saver1.restore(sess1, FLAGS.img_model_path)

    log_string(str(datetime.now()))

    # To collect statistics
    total_correct = 0
    total_seen = 0
    num_batches = 0
    tp = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
    fp = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
    fn = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}

    proposal_boxes = []
    gt_boxes = []
    nms_indices = []

    while(True):
        batch_data, is_last_batch = dataset.get_next_batch(BATCH_SIZE, need_id=True)

        start = datetime.now()

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

        # segmentaion with image
        seg_pls = img_seg_net.placeholders
        img_seg_logits, full_img_seg = sess1.run([seg_softmax, full_seg], feed_dict={
            seg_pls['pointclouds']: batch_data['pointcloud'],
            seg_pls['img_inputs']: batch_data['images'],
            seg_pls['calib']: batch_data['calib'],
            seg_pls['seg_labels']: batch_data['seg_label'],
            #seg_pls['is_training_pl']: is_training
        })
        # convert to binary segmentation
        img_seg_binary = np.zeros((BATCH_SIZE, NUM_POINT, 2))
        img_seg_binary[...,0] = img_seg_logits[...,0]
        img_seg_binary[...,1] = np.sum(img_seg_logits[...,1:], axis=-1)
        img_seg_binary *= np.array([0, 1]) # weights
        feed_dict[pls['img_seg_softmax']] = img_seg_binary
        '''
        # label to one_hot
        targets = batch_data['seg_label']
        img_seg_logits = np.eye(NUM_SEG_CLASSES)[targets]
        #img_seg_logits *= np.array([2, 2, 2, 2]) # weights
        feed_dict[pls['img_seg_softmax']] = img_seg_logits
        '''

        logits_val, indices_val, centers_val, angles_val, sizes_val, corners_val, ind_val, scores_val \
        = sess.run([
            end_points['foreground_logits'], end_points['fg_point_indices'],
            box_center, box_angle, box_size, end_points['proposal_boxes'],
            end_points['nms_indices'], end_points['proposal_scores']], feed_dict=feed_dict)
        print('inference time: ', datetime.now() - start)
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        num_batches += 1
        for c in ['Car', 'Pedestrian', 'Cyclist']:
            one_hot_class = g_type2onehotclass[c]
            tp[c] += np.sum(np.logical_and(preds_val == batch_data['seg_label'], batch_data['seg_label'] == one_hot_class))
            fp[c] += np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] != one_hot_class))
            fn[c] += np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] == one_hot_class))
        # results
        for i in range(BATCH_SIZE):
            proposal_boxes.append(corners_val[i])
            gt_boxes.append(batch_data['gt_boxes'][i])
            nms_indices.append(ind_val[i])
            frame_data = {
                'frame_id': batch_data['ids'][i],
                'segmentation': preds_val[i], # only point seg
                'fg_indices': indices_val[i], # ensemble with img seg
                'centers': centers_val[i],
                'angles': angles_val[i],
                'sizes': sizes_val[i],
                'proposal_boxes': corners_val[i],
                'nms_indices': ind_val[i],
                'scores': scores_val[i],
                'pc_choices': batch_data['pc_choice'][i]
            }
            # save frame data
            if save_result:
                with open(os.path.join('./rcnn_data_'+split, batch_data['ids'][i]+'.pkl'), 'wb') as fout:
                    pickle.dump(frame_data, fout)
                np.save(os.path.join('./rcnn_data_'+split, batch_data['ids'][i]+'_seg.npy'), full_img_seg[i])
        if is_last_batch:
            break

    log_string('saved prediction')

    dataset.stop_loading()
    produce_thread.join()

    '''
    all_indices = np.tile(np.arange(1024), (len(proposal_boxes),))
    iou2d, iou3d = compute_box3d_iou(proposal_boxes, point_gt_boxes, all_indices)
    print('IOU2d: ', np.mean(iou2d))
    print('IOU3d: ', np.mean(iou3d))
    '''
    if split in ['train', 'val']:
        recall = compute_proposal_recall(proposal_boxes, gt_boxes, nms_indices)
        print('Average recall: ', recall)
        print(tp, fp, fn)
        for c in ['Car', 'Pedestrian', 'Cyclist']:
            if (tp[c]+fn[c] == 0) or (tp[c]+fp[c]) == 0:
                continue
            print(c + ' segmentation recall: %f'% \
                (float(tp[c])/(tp[c]+fn[c])))
            print(c + ' segmentation precision: %f'% \
                (float(tp[c])/(tp[c]+fp[c])))

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    #TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val', types=['Car'], difficulties=[1])
    test(SPLIT, save_result=True)
