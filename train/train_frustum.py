''' Training Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from frustum_model_util import NUM_SEG_CLASSES, NUM_OBJ_CLASSES, g_type2onehotclass, NUM_CHANNEL
from frustum_dataset import FrustumDataset, Sample
import provider
from frustum_pointnets_v2 import FrustumPointNet

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
parser.add_argument('--pos_ratio', type=float, default=0.5, help='Positive proposal ratio')
parser.add_argument('--train_cls_only', type=int, default=0, help='Train classification only')
parser.add_argument('--train_reg_only', type=int, default=0, help='Train box regression only')
parser.add_argument('--use_gt_prop', type=int, default=0, help='Use label to generate proposal or not')
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
# NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# load data set in background thread, remember to join data_loading_thread somewhere
TRAIN_DATASET = FrustumDataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', BATCH_SIZE, 'train',
             data_dir='./rcnn_data_train', is_training=True,
             augmentX=5, random_shift=True, rotate_to_center=True, random_flip=True, use_gt_prop=FLAGS.use_gt_prop)
TEST_DATASET = FrustumDataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', BATCH_SIZE, 'val',
             data_dir='./rcnn_data_val', is_training=True,
             augmentX=1, random_shift=False, rotate_to_center=True, random_flip=False, use_gt_prop=FLAGS.use_gt_prop)
train_loading_thread = Thread(target=TRAIN_DATASET.load, args=(FLAGS.pos_ratio,))
val_loading_thread = Thread(target=TEST_DATASET.load, args=(FLAGS.pos_ratio,))
train_loading_thread.start()
val_loading_thread.start()

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

def train():
    ''' Main function for training and simple evaluation. '''
    best_val_loss = float('inf')
    best_avg_cls_acc = 0
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and losses
            frustum_pointnet = FrustumPointNet(BATCH_SIZE, NUM_POINT, bn_decay)
            end_points = frustum_pointnet.end_points
            pls = frustum_pointnet.placeholders
            loss, loss_endpoints = frustum_pointnet.get_loss()
            tf.summary.scalar('loss', loss)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)

            # Write summaries of bounding box IoU and segmentation accuracies
            iou2ds, iou3ds = tf.py_func(provider.compute_box3d_iou, [\
                end_points['center'], \
                end_points['heading_scores'], end_points['heading_residuals'], \
                end_points['size_scores'], end_points['size_residuals'], \
                pls['centers'], \
                pls['heading_class_label'], pls['heading_residual_label'], \
                pls['size_class_label'], pls['size_residual_label']], \
                [tf.float32, tf.float32])
            end_points['iou2ds'] = iou2ds
            end_points['iou3ds'] = iou3ds
            tf.summary.scalar('iou_2d', tf.reduce_mean(iou2ds))
            tf.summary.scalar('iou_3d', tf.reduce_mean(iou3ds))

            correct = tf.equal(tf.argmax(end_points['mask_logits'], 2),
                tf.to_int64(pls['seg_labels']))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / \
                float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('segmentation accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                    momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            if FLAGS.train_cls_only:
                # var_list = tf.trainable_variables('cls_')
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='proposal_classification')
                loss = loss_endpoints['cls_loss']
            elif FLAGS.train_reg_only:
                # tvars = tf.trainable_variables()
                # var_list = [var for var in tvars if 'cls_' not in var.name]
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='box_regression')
                loss = loss_endpoints['box_loss']
            else:
                var_list = tf.trainable_variables()

            # Note: when training, the moving_mean and moving_variance need to be updated.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=batch, var_list=var_list)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
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

        ops = {'pointclouds_pl': pls['pointclouds'],
               'img_seg_map_pl': pls['img_seg_map'],
               'prop_box_pl': pls['prop_box'],
               'calib_pl': pls['calib'],
               'cls_label_pl': pls['cls_label'],
               'ious_pl': pls['ious'],
               'labels_pl': pls['seg_labels'],
               'centers_pl': pls['centers'],
               'heading_class_label_pl': pls['heading_class_label'],
               'heading_residual_label_pl': pls['heading_residual_label'],
               'size_class_label_pl': pls['size_class_label'],
               'size_residual_label_pl': pls['size_residual_label'],
               'is_training_pl': pls['is_training'],
               'logits': end_points['mask_logits'],
               'cls_logits': end_points['cls_logits'],
               'centers_pred': end_points['center'],
               'loss': loss,
               'loss_endpoints': loss_endpoints,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            # Save the variables to disk.
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            #     log_string("Model saved in file: {0}, val_loss: {1}".format(save_path, val_loss))
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt.%03d" % epoch))
            log_string("Model saved in file: {0}".format(save_path))
            val_loss, avg_cls_acc, estimate_acc = eval_one_epoch(sess, ops, test_writer)
        train_loading_thread.stop()
        val_loading_thread.stop()

def train_one_epoch(sess, ops, train_writer, idxs_to_use=None):
    ''' Training for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops
    '''
    is_training = True
    log_string(str(datetime.now()))

    # To collect statistics
    total_cls_correct = 0
    total_cls_seen = 0
    total_correct = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_seen = 0
    loss_sum = 0
    total_obj_sample = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Training with batches
    # for batch_idx in range(num_batches):
    batch_idx = 0
    while(True):
        batch_data, is_last_batch = TRAIN_DATASET.get_next_batch()
        # FIXME: will discard last batch if it has less samples than batch size
        if len(batch_data['ids']) != BATCH_SIZE:
            break

        feed_dict = {ops['pointclouds_pl']: batch_data['pointcloud'],
                     ops['img_seg_map_pl']: batch_data['img_seg_map'],
                     ops['prop_box_pl']: batch_data['prop_box'],
                     ops['calib_pl']: batch_data['calib'],
                     ops['cls_label_pl']: batch_data['cls_label'],
                     ops['ious_pl']: batch_data['ious'],
                     ops['labels_pl']: batch_data['seg_label'],
                     ops['centers_pl']: batch_data['center'],
                     ops['heading_class_label_pl']: batch_data['heading_class'],
                     ops['heading_residual_label_pl']: batch_data['heading_residual'],
                     ops['size_class_label_pl']: batch_data['size_class'],
                     ops['size_residual_label_pl']: batch_data['size_residual'],
                     ops['is_training_pl']: is_training,}

        summary, step, _, loss_val, cls_logits_val, logits_val, centers_pred_val, \
        iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'],
                ops['cls_logits'], ops['logits'], ops['centers_pred'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        # classification acc
        cls_preds_val = np.argmax(cls_logits_val, 1)
        cls_correct = np.sum(cls_preds_val == batch_data['cls_label'])
        tp = np.sum(np.logical_and(cls_preds_val == batch_data['cls_label'], batch_data['cls_label'] != g_type2onehotclass['NonObject']))
        fp = np.sum(np.logical_and(cls_preds_val != batch_data['cls_label'], batch_data['cls_label'] == g_type2onehotclass['NonObject']))
        fn = np.sum(np.logical_and(cls_preds_val != batch_data['cls_label'], batch_data['cls_label'] != g_type2onehotclass['NonObject']))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_cls_correct += cls_correct
        total_cls_seen += BATCH_SIZE
        # only calculate seg acc and regression performance with object labels
        obj_mask = batch_data['cls_label'] != g_type2onehotclass['NonObject']
        obj_sample_num = np.sum(obj_mask)
        total_obj_sample += obj_sample_num
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val[obj_mask] == batch_data['seg_label'][obj_mask])
        total_correct += correct
        total_seen += (obj_sample_num*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds[obj_mask])
        iou3ds_sum += np.sum(iou3ds[obj_mask])
        iou3d_correct_cnt += np.sum(iou3ds[obj_mask]>=0.7)

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d --' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('classification accuracy: %f' % \
                (total_cls_correct / float(total_cls_seen)))
            if total_tp+total_fn > 0:
                log_string('recall: %f'% \
                    (float(total_tp)/(total_tp+total_fn)))
            if total_tp+total_fp > 0:
                log_string('precision: %f'% \
                    (float(total_tp)/(total_tp+total_fp)))
            if total_seen > 0:
                log_string('segmentation accuracy: %f' % \
                    (total_correct / float(total_seen)))
            if total_obj_sample > 0:
                log_string('box IoU (ground/3D): %f / %f' % \
                    (iou2ds_sum / float(total_obj_sample), iou3ds_sum / float(total_obj_sample)))
                log_string('box estimation accuracy (IoU=0.7): %f' % \
                    (float(iou3d_correct_cnt)/float(total_obj_sample)))
            total_cls_correct = 0
            total_correct = 0
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_cls_seen = 0
            total_seen = 0
            total_obj_sample = 0
            loss_sum = 0
            iou2ds_sum = 0
            iou3ds_sum = 0
            iou3d_correct_cnt = 0
        if is_last_batch:
            break
        batch_idx += 1


def eval_one_epoch(sess, ops, test_writer):
    ''' Simple evaluation for one epoch on the frustum dataset.
    ops is dict mapping from string to tf ops """
    '''
    global EPOCH_CNT
    is_training = False
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    # test_idxs = np.arange(0, len(TEST_DATASET))
    # num_batches = len(TEST_DATASET)/BATCH_SIZE

    # To collect statistics
    total_cls_correct = 0
    total_cls_seen = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_seen_class = [0 for _ in range(NUM_OBJ_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_OBJ_CLASSES)]
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_obj_sample = 0
    iou2ds_sum = 0
    iou3ds_sum = 0
    iou3d_correct_cnt = 0

    # Simple evaluation with batches
    # for batch_idx in range(num_batches):
    num_batches = 0
    while(True):
        batch_data, is_last_batch = TEST_DATASET.get_next_batch()

        # FIXME: will discard last batch if it has less samples than batch size
        if len(batch_data['ids']) != BATCH_SIZE:
            break

        feed_dict = {ops['pointclouds_pl']: batch_data['pointcloud'],
                     ops['img_seg_map_pl']: batch_data['img_seg_map'],
                     ops['prop_box_pl']: batch_data['prop_box'],
                     ops['calib_pl']: batch_data['calib'],
                     ops['cls_label_pl']: batch_data['cls_label'],
                     ops['ious_pl']: batch_data['ious'],
                     ops['labels_pl']: batch_data['seg_label'],
                     ops['centers_pl']: batch_data['center'],
                     ops['heading_class_label_pl']: batch_data['heading_class'],
                     ops['heading_residual_label_pl']: batch_data['heading_residual'],
                     ops['size_class_label_pl']: batch_data['size_class'],
                     ops['size_residual_label_pl']: batch_data['size_residual'],
                     ops['is_training_pl']: is_training or FLAGS.train_reg_only}

        summary, step, loss_val, loss_endpoints, cls_logits_val, logits_val, iou2ds, iou3ds = \
            sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['loss_endpoints'], ops['cls_logits'], ops['logits'],
                ops['end_points']['iou2ds'], ops['end_points']['iou3ds']],
                feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        if np.isnan(loss_val):
            print('nan loss in batch: ', num_batches)
            print('loss_endpoints: ', loss_endpoints)

        # classification acc
        cls_preds_val = np.argmax(cls_logits_val, 1)
        cls_correct = np.sum(cls_preds_val == batch_data['cls_label'])
        tp = np.sum(np.logical_and(cls_preds_val == batch_data['cls_label'], batch_data['cls_label'] != g_type2onehotclass['NonObject']))
        fp = np.sum(np.logical_and(cls_preds_val != batch_data['cls_label'], batch_data['cls_label'] == g_type2onehotclass['NonObject']))
        fn = np.sum(np.logical_and(cls_preds_val != batch_data['cls_label'], batch_data['cls_label'] != g_type2onehotclass['NonObject']))
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_cls_correct += cls_correct
        total_cls_seen += BATCH_SIZE
        for l in range(NUM_OBJ_CLASSES):
            total_seen_class[l] += np.sum(batch_data['cls_label']==l)
            total_correct_class[l] += (np.sum((cls_preds_val==l) & (batch_data['cls_label']==l)))

        # only calculate seg acc and regression performance with object labels
        obj_mask = batch_data['cls_label'] != g_type2onehotclass['NonObject']
        obj_sample_num = np.sum(obj_mask)
        total_obj_sample += obj_sample_num
        # segmentation acc
        preds_val = np.argmax(logits_val, 2)
        correct = np.sum(preds_val[obj_mask] == batch_data['seg_label'][obj_mask])
        total_correct += correct
        total_seen += (obj_sample_num*NUM_POINT)
        loss_sum += loss_val
        iou2ds_sum += np.sum(iou2ds[obj_mask])
        iou3ds_sum += np.sum(iou3ds[obj_mask])
        iou3d_correct_cnt += np.sum(iou3ds[obj_mask]>=0.7)

        num_batches += 1
        if is_last_batch:
            break

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('classification accuracy: %f' % \
        (total_cls_correct / float(total_cls_seen)))
    log_string('eval segmentation accuracy: %f'% \
        (total_correct / float(total_seen)))
    log_string('recall: %f'% \
        (float(total_tp)/(total_tp+total_fn)))
    log_string('precision: %f'% \
        (float(total_tp)/(total_tp+total_fp)))
    avg_cls_acc = np.mean(np.array(total_correct_class) / \
        np.array(total_seen_class,dtype=np.float))
    log_string('eval classification avg class acc: %f' % avg_cls_acc)
    if total_obj_sample > 0:
        log_string('eval box IoU (ground/3D): %f / %f' % \
            (iou2ds_sum / float(total_obj_sample), iou3ds_sum / \
                float(total_obj_sample)))
        box_estimation_acc = float(iou3d_correct_cnt)/float(total_obj_sample)
        log_string('eval box estimation accuracy (IoU=0.7): %f' % box_estimation_acc)
    else:
        box_estimation_acc = 0
    mean_loss = loss_sum / float(num_batches)
    EPOCH_CNT += 1
    return mean_loss, avg_cls_acc, box_estimation_acc


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
