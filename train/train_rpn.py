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
from rpn_dataset import Dataset
from model_util import NUM_FG_POINT
from rpn import RPN, NUM_SEG_CLASSES
import train_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [default: 16384]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.002]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

if FLAGS.no_intensity:
    NUM_CHANNEL = 3
else:
    NUM_CHANNEL = 4

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


TRAIN_DATASET = Dataset(NUM_POINT, NUM_CHANNEL, '/data/ssd/public/jlliu/Kitti/object', 'train', is_training=True, use_aug_scene=True)
# data loading threads
train_produce_thread = Thread(target=TRAIN_DATASET.load, args=(True,))
train_produce_thread.start()

def train():
    ''' Main function for training and simple evaluation. '''

    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            # is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and losses
            rpn_model = RPN(BATCH_SIZE, NUM_POINT, num_channel=NUM_CHANNEL, bn_decay=bn_decay, is_training=True)
            placeholders = rpn_model.placeholders
            end_points = rpn_model.end_points
            loss, loss_endpoints = rpn_model.get_loss()

            iou2ds, iou3ds = tf.py_func(train_util.compute_box3d_iou, [
                    end_points['proposal_boxes'],
                    end_points['gt_box_of_point'],
                    end_points['nms_indices']
                ], [tf.float32, tf.float32])
            end_points['iou2ds'] = iou2ds
            end_points['iou3ds'] = iou3ds

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                    momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # Note: when training, the moving_mean and moving_variance need to be updated.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #train_op = optimizer.minimize(loss, global_step=batch)
                train_op = slim.learning.create_train_op(
                    loss,
                    optimizer,
                    clip_gradient_norm=1.0,
                    global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if FLAGS.restore_model_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess, FLAGS.restore_model_path)

        ops = {
            'loss': loss,
            'train_op': train_op,
            'step': batch,
            'merged': merged,
            'loss_endpoints': loss_endpoints,
            'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            # eval iou and recall is slow
            #eval_iou_recall = epoch > 10
            #eval_iou_recall = False
            eval_iou_recall = epoch % 2 == 0
            train_one_epoch(sess, ops, placeholders, train_writer, eval_iou_recall)
            # Save the variables to disk.
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            #     log_string("Model saved in file: {0}, val_loss: {1}".format(save_path, val_loss))
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt.%03d" % epoch))
            log_string("Model saved in file: {0}".format(save_path))
            val_loss = eval_one_epoch(sess, ops, placeholders, test_writer, True)
    TRAIN_DATASET.stop_loading()
    train_produce_thread.join()


def train_one_epoch(sess, ops, pls, train_writer, more=False):
    is_training = True
    log_string(str(datetime.now()))

    # To collect statistics
    total_correct = 0
    total_seen = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    loss_sum = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    total_nms = 0
    total_proposal_recall = 0

    # Training with batches
    batch_idx = 0
    while(True):
        batch_data, is_last_batch = TRAIN_DATASET.get_next_batch(BATCH_SIZE)

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
            pls['img_seg_softmax']: np.zeros((BATCH_SIZE, NUM_POINT, NUM_SEG_CLASSES)),
            pls['is_training_pl']: is_training,
        }
        if more:
            summary, step, loss_val, _, logits_val, iou2ds, iou3ds, proposal_boxes, nms_indices \
            = sess.run([
                ops['merged'], ops['step'], ops['loss'], ops['train_op'],
                ops['end_points']['foreground_logits'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],
                ops['end_points']['proposal_boxes'], ops['end_points']['nms_indices']], feed_dict=feed_dict)
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            total_nms += len(iou2ds)
            # average on each frame
            proposal_recall = train_util.compute_proposal_recall(proposal_boxes, batch_data['gt_boxes'], nms_indices)
            total_proposal_recall += proposal_recall * BATCH_SIZE
        else:
            summary, step, loss_val, _, logits_val = sess.run([
                ops['merged'], ops['step'], ops['loss'], ops['train_op'],
                ops['end_points']['foreground_logits']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_data['seg_label'])
        tp = np.sum(np.logical_and(preds_val == batch_data['seg_label'], batch_data['seg_label'] != 0))
        fp = np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] == 0))
        fn = np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] != 0))
        total_correct += correct
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_seen += NUM_POINT * BATCH_SIZE
        loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            sample_num = 10 * BATCH_SIZE
            log_string(' -- %03d --' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / sample_num))
            if total_seen > 0:
                log_string('segmentation accuracy: %f' % \
                    (total_correct / float(total_seen)))
            if total_tp+total_fn > 0 and total_tp+total_fp > 0:
                log_string('segmentation recall: %f'% \
                    (float(total_tp)/(total_tp+total_fn)))
                log_string('segmentation precision: %f'% \
                    (float(total_tp)/(total_tp+total_fp)))
            if more:
                log_string('box IoU (ground/3D): %f / %f' % \
                    (iou2ds_sum / float(total_nms), iou3ds_sum / float(total_nms)))
                log_string('proposal recall: %f' % (float(total_proposal_recall) / sample_num))
            if np.isnan(loss_sum):
                loss_endpoints = sess.run(ops['loss_endpoints'], feed_dict=feed_dict)
                print('loss_endpoints: ', loss_endpoints)
            total_correct = 0
            total_seen = 0
            total_tp = 0
            total_fp = 0
            total_fn = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            total_nms = 0
            total_proposal_recall = 0
        if is_last_batch:
            break
        batch_idx += 1



def eval_one_epoch(sess, ops, pls, test_writer, more=False):
    TEST_DATASET = Dataset(NUM_POINT, NUM_CHANNEL, '/data/ssd/public/jlliu/Kitti/object', 'val', is_training=True)
    test_produce_thread = Thread(target=TEST_DATASET.load, args=(False,))
    test_produce_thread.start()

    global EPOCH_CNT
    is_training = False
    #is_training = True
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    # To collect statistics
    total_correct = 0
    total_seen = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    loss_sum = 0
    num_samples = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    total_nms = 0
    total_proposal_recall = 0

    while(True):
        batch_data, is_last_batch = TEST_DATASET.get_next_batch(BATCH_SIZE)

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
            pls['img_seg_softmax']: np.zeros((BATCH_SIZE, NUM_POINT, NUM_SEG_CLASSES)),
            pls['is_training_pl']: is_training,
        }

        if more:
            summary, step, loss_val, logits_val, iou2ds, iou3ds, proposal_boxes, nms_indices \
            = sess.run([
                ops['merged'], ops['step'], ops['loss'],
                ops['end_points']['foreground_logits'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds'],
                ops['end_points']['proposal_boxes'], ops['end_points']['nms_indices']], feed_dict=feed_dict)
                #feed_dict=feed_dict)
            iou2ds_sum += np.sum(iou2ds)
            iou3ds_sum += np.sum(iou3ds)
            total_nms += len(iou2ds)
            # average on each frame
            proposal_recall = train_util.compute_proposal_recall(proposal_boxes, batch_data['gt_boxes'], nms_indices)
            total_proposal_recall += proposal_recall * BATCH_SIZE
        else:
            summary, step, loss_val, logits_val = sess.run([
                ops['merged'], ops['step'], ops['loss'],
                ops['end_points']['foreground_logits']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val == batch_data['seg_label'])
        tp = np.sum(np.logical_and(preds_val == batch_data['seg_label'], batch_data['seg_label'] != 0))
        fp = np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] == 0))
        fn = np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] != 0))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_correct += correct
        total_seen += NUM_POINT * BATCH_SIZE
        loss_sum += loss_val
        num_samples += BATCH_SIZE
        if is_last_batch:
            break

    log_string('eval mean loss: %f' % (loss_sum / float(num_samples)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    if total_tp+total_fn > 0 and total_tp+total_fp > 0:
        log_string('eval segmentation recall: %f'% \
            (float(total_tp)/(total_tp+total_fn)))
        log_string('eval segmentation precision: %f'% \
            (float(total_tp)/(total_tp+total_fp)))
    if more:
        log_string('eval box IoU (ground/3D): %f / %f' % \
            (iou2ds_sum / float(total_nms), iou3ds_sum / float(total_nms)))
        log_string('eval proposal recall: %f' % (float(total_proposal_recall) / num_samples))
    EPOCH_CNT += 1

    TEST_DATASET.stop_loading()
    test_produce_thread.join()

    return loss_sum / float(num_samples)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
