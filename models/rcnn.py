import sys
import os
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from box_encoder import NUM_OBJ_CLASSES, type_mean_size, NUM_SIZE_CLUSTER
from box_encoder import BoxEncoder
from model_util import huber_loss, get_box3d_corners_helper
import projection
from img_vgg_pyramid import ImgVggPyr
from collections import namedtuple

NUM_HEADING_BIN = 9
NUM_CENTER_BIN = 6
CENTER_SEARCH_RANGE = 1.5
HEADING_SEARCH_RANGE = 0.25*np.pi

class RCNN(object):
    def __init__(self, batch_size, num_point, num_channel=133, bn_decay=None, is_training=True):
        self.batch_size = batch_size
        self.num_point = num_point
        self.num_channel = num_channel
        self.bn_decay = bn_decay
        self.is_training = is_training
        self.end_points = {}
        self.placeholders = self.get_placeholders()
        self.box_encoder = BoxEncoder(CENTER_SEARCH_RANGE, NUM_CENTER_BIN, HEADING_SEARCH_RANGE, NUM_HEADING_BIN)
        self.build()

    def get_placeholders(self):
        batch_size = self.batch_size
        num_point = self.num_point
        num_channel = self.num_channel
        return {
            'pointclouds': tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channel)),
            'proposal_boxes': tf.placeholder(tf.float32, shape=(batch_size, 7)),
            'class_labels': tf.placeholder(tf.int32, shape=(batch_size,)),
            'center_bin_x_labels': tf.placeholder(tf.int32, shape=(batch_size,)),
            'center_bin_z_labels': tf.placeholder(tf.int32, shape=(batch_size,)),
            'center_x_res_labels': tf.placeholder(tf.float32, shape=(batch_size,)),
            'center_z_res_labels': tf.placeholder(tf.float32, shape=(batch_size,)),
            'center_y_res_labels': tf.placeholder(tf.float32, shape=(batch_size,)),
            'heading_bin_labels': tf.placeholder(tf.int32, shape=(batch_size,)),
            'heading_res_labels': tf.placeholder(tf.float32, shape=(batch_size,)),
            'size_class_labels': tf.placeholder(tf.int32, shape=(batch_size,)),
            'size_res_labels': tf.placeholder(tf.float32, shape=(batch_size, 3)),
            'gt_box_of_prop': tf.placeholder(tf.float32, shape=(batch_size, 8, 3)),
            'img_inputs': tf.placeholder(tf.float32, shape=(batch_size, 360, 1200, 3)),
            'calib': tf.placeholder(tf.float32, shape=(batch_size, 3, 4)),
            'train_regression': tf.placeholder(tf.bool, shape=(batch_size,)),
            'img_seg_map': tf.placeholder(tf.float32, shape=(batch_size, 360, 1200, 4)),
            'is_training_pl': tf.placeholder(tf.bool, shape=())
        }

    def build(self):
        point_cloud = self.placeholders['pointclouds']
        is_training = self.placeholders['is_training_pl']
        batch_size = self.batch_size
        # image
        '''
        seg_softmax = self.placeholders['img_seg_map']
        seg_pred = tf.expand_dims(tf.argmax(seg_softmax, axis=-1), axis=-1)
        self._img_pixel_size = np.asarray([360, 1200])
        box2d_corners, box2d_corners_norm = projection.tf_project_to_image_space(
            self.placeholders['proposal_boxes'],
            self.placeholders['calib'], self._img_pixel_size)
        # y1, x1, y2, x2
        box2d_corners_norm_reorder = tf.stack([
            tf.gather(box2d_corners_norm, 1, axis=-1),
            tf.gather(box2d_corners_norm, 0, axis=-1),
            tf.gather(box2d_corners_norm, 3, axis=-1),
            tf.gather(box2d_corners_norm, 2, axis=-1),
        ], axis=-1)
        img_rois = tf.image.crop_and_resize(
            seg_softmax,
            #seg_pred,
            box2d_corners_norm_reorder,
            tf.range(0, batch_size),
            [16,16])
        self.end_points['img_rois'] = img_rois
        self.end_points['box2d_corners_norm_reorder'] = box2d_corners_norm_reorder
        '''

        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        if self.num_channel > 3:
            l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,self.num_channel-3])
        else:
            l0_points = None
        # Set abstraction layers
        l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points,
            npoint=128, radius=0.2, nsample=64, mlp=[128,128,128],
            mlp2=None, group_all=False, is_training=is_training, bn_decay=self.bn_decay,
            scope='rcnn-sa1', bn=True)
        l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points,
            npoint=32, radius=0.4, nsample=64, mlp=[128,128,256],
            mlp2=None, group_all=False, is_training=is_training, bn_decay=self.bn_decay,
            scope='rcnn-sa2', bn=True)
        l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
            npoint=-1, radius=100, nsample=64, mlp=[256,256,512],
            mlp2=None, group_all=True, is_training=is_training, bn_decay=self.bn_decay,
            scope='rcnn-sa3', bn=True)

        point_feats = l3_points

        # Classification
        cls_net = tf_util.conv1d(point_feats, 256, 1, padding='VALID', bn=True,
            is_training=is_training, scope='rcnn-cls-fc1', bn_decay=self.bn_decay)
        # cls_net = tf_util.dropout(cls_net, keep_prob=0.5,
        #     is_training=is_training, scope='rcnn-cls-dp')
        cls_net = tf_util.conv1d(cls_net, 256, 1, padding='VALID', bn=True,
            is_training=is_training, scope='rcnn-cls-fc2', bn_decay=self.bn_decay)
        cls_out = tf_util.conv1d(cls_net, NUM_OBJ_CLASSES, 1,
            padding='VALID', activation_fn=None, scope='conv1d-fc2')
        cls_out = tf.squeeze(cls_out, axis=1)
        self.end_points['cls_logits'] = cls_out

        # Box estimation
        cls_label_pred = tf.argmax(tf.nn.softmax(cls_net), axis=-1)
        one_hot_pred = tf.one_hot(cls_label_pred, NUM_OBJ_CLASSES, axis=-1) # (B, 1, NUM_OBJ_CLASSES)
        one_hot_gt = tf.one_hot(self.placeholders['class_labels'], NUM_OBJ_CLASSES, axis=-1) # (B, NUM_OBJ_CLASSES)
        one_hot_gt = tf.expand_dims(one_hot_gt, axis=1)
        one_hot_vec = tf.cond(is_training, lambda: one_hot_gt, lambda: one_hot_pred)
        one_hot_vec.set_shape([batch_size, 1, NUM_OBJ_CLASSES])
        est_intput = tf.concat([point_feats, one_hot_vec], axis=-1)
        box_net = tf_util.conv1d(est_intput, 256, 1, padding='VALID', bn=True,
            is_training=is_training, scope='rcnn-box-fc1', bn_decay=self.bn_decay)
        # cls_net = tf_util.dropout(cls_net, keep_prob=0.5,
        #     is_training=is_training, scope='rcnn-cls-dp')
        box_net = tf_util.conv1d(box_net, 256, 1, padding='VALID', bn=True,
            is_training=is_training, scope='rcnn-box-fc2', bn_decay=self.bn_decay)
        # The first NUM_CENTER_BIN*2*2: CENTER_BIN class scores and bin residuals for (x,z)
        # next 1: center residual for y
        # next NUM_HEADING_BIN*2: heading bin class scores and residuals
        # next NUM_SIZE_CLUSTER*4: size cluster class scores and residuals(l,w,h)
        box_out = tf_util.conv1d(box_net, NUM_CENTER_BIN*2*2+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, 1,
            padding='VALID', activation_fn=None, scope='rcnn-box-out')
        box_out = tf.squeeze(box_out, axis=1)
        self.parse_output_to_tensors(box_out)
        self.get_output_boxes()

    def parse_output_to_tensors(self, output):
        ''' Parse batch output to separate tensors (added to end_points)'''
        batch_size = self.batch_size
        # objectness and center
        #end_points['objectness'] = tf.slice(output, [0,0,0], [-1,-1,2])
        center_x_scores = tf.slice(output, [0,0], [-1,NUM_CENTER_BIN])
        center_x_residuals_normalized = tf.slice(output, [0,NUM_CENTER_BIN],
            [-1,NUM_CENTER_BIN])
        self.end_points['center_x_scores'] = center_x_scores # (B,NUM_CENTER_BIN)
        self.end_points['center_x_residuals_normalized'] = \
            center_x_residuals_normalized # (B,NUM_CENTER_BIN)
        center_z_scores = tf.slice(output, [0,NUM_CENTER_BIN*2], [-1,NUM_CENTER_BIN])
        center_z_residuals_normalized = tf.slice(output, [0,NUM_CENTER_BIN*3],
            [-1,NUM_CENTER_BIN])
        self.end_points['center_z_scores'] = center_z_scores # (B,NUM_CENTER_BIN)
        self.end_points['center_z_residuals_normalized'] = \
            center_z_residuals_normalized # (B,NUM_CENTER_BIN)
        self.end_points['center_y_residuals'] = tf.slice(output, [0,NUM_CENTER_BIN*4], [-1,1])
        # heading
        heading_scores = tf.slice(output, [0,NUM_CENTER_BIN*4+1], [-1,NUM_HEADING_BIN])
        heading_residuals_normalized = tf.slice(output, [0,NUM_CENTER_BIN*4+1+NUM_HEADING_BIN],
            [-1,NUM_HEADING_BIN])
        self.end_points['heading_scores'] = heading_scores # (B,NUM_HEADING_BIN)
        self.end_points['heading_residuals_normalized'] = heading_residuals_normalized # (B,NUM_HEADING_BIN)
        # end_points['heading_residuals'] = \
        #     heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN
        # size
        size_scores = tf.slice(output, [0,NUM_CENTER_BIN*4+1+NUM_HEADING_BIN*2],
            [-1,NUM_SIZE_CLUSTER]) # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = tf.slice(output,
            [0,NUM_CENTER_BIN*4+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER], [-1,NUM_SIZE_CLUSTER*3])
        size_residuals_normalized = tf.reshape(size_residuals_normalized,
            [batch_size, NUM_SIZE_CLUSTER, 3])
        self.end_points['size_scores'] = size_scores
        self.end_points['size_residuals_normalized'] = size_residuals_normalized
        # end_points['size_residuals'] = size_residuals_normalized * \
        #     tf.expand_dims(tf.constant(type_mean_size, dtype=tf.float32), 0)
        return self.end_points

    def get_output_boxes(self):
        end_points = {}
        # adapt the dimension
        for k in ['center_x_scores', 'center_x_residuals_normalized',
            'center_z_scores', 'center_z_residuals_normalized',
            'center_y_residuals', 'heading_scores', 'heading_residuals_normalized',
            'size_scores', 'size_residuals_normalized']:
            end_points[k] = tf.expand_dims(self.end_points[k], axis=1)
        box_center, box_angle, box_size = self.box_encoder.tf_decode(end_points)
        box_center = tf.squeeze(box_center, axis=1)
        box_center = box_center + tf.slice(self.placeholders['proposal_boxes'], [0,0], [-1,3])
        box_angle = tf.squeeze(box_angle, axis=1)
        box_angle += tf.gather(self.placeholders['proposal_boxes'], 6, axis=-1) # resotre absoluate angle
        box_size = tf.squeeze(box_size, axis=1)
        self.end_points['box_center'] = box_center
        self.end_points['box_angle'] = box_angle
        self.end_points['box_size'] = box_size
        corners_3d = get_box3d_corners_helper(box_center, box_angle, box_size)
        self.end_points['box_corners'] = corners_3d
        # box score
        seg_scores = tf.reduce_max(tf.nn.softmax(self.end_points['cls_logits']), axis=-1) # (B,)
        bin_x_scores = tf.reduce_max(tf.nn.softmax(self.end_points['center_x_scores']), axis=-1) # (B,M)
        bin_z_scores = tf.reduce_max(tf.nn.softmax(self.end_points['center_z_scores']), axis=-1) # (B,M)
        heading_scores = tf.reduce_max(tf.nn.softmax(self.end_points['heading_scores']), axis=-1) # (B,M)
        size_scores = tf.reduce_max(tf.nn.softmax(self.end_points['size_scores']), axis=-1) # (B,M)
        # confidence = seg_scores + bin_x_scores + bin_z_scores + heading_scores + size_scores
        confidence = seg_scores * bin_x_scores * bin_z_scores * heading_scores * size_scores
        self.end_points['box_score'] = confidence
        return corners_3d

    def get_loss(self):
        end_points = self.end_points
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=end_points['cls_logits'], labels=self.placeholders['class_labels']))
        tf.summary.scalar('classification loss', cls_loss)
        # is_obj_mask = tf.to_float(tf.not_equal(self.placeholders['class_labels'], 0))
        train_reg_mask = tf.to_float(self.placeholders['train_regression'])
        center_x_cls_loss = tf.reduce_mean(train_reg_mask*tf.nn.sparse_softmax_cross_entropy_with_logits(\
           logits=end_points['center_x_scores'], labels=self.placeholders['center_bin_x_labels']))
        center_z_cls_loss = tf.reduce_mean(train_reg_mask*tf.nn.sparse_softmax_cross_entropy_with_logits(\
           logits=end_points['center_z_scores'], labels=self.placeholders['center_bin_z_labels']))
        bin_x_onehot = tf.one_hot(self.placeholders['center_bin_x_labels'],
            depth=NUM_CENTER_BIN,
            on_value=1, off_value=0, axis=-1) # BxNUM_CENTER_BIN
        # NOTICE: labels['center_x_residuals'] is already normalized
        center_x_residuals_normalized = tf.reduce_sum(end_points['center_x_residuals_normalized']*tf.to_float(bin_x_onehot), axis=-1) # B
        center_x_residuals_dist = tf.norm(self.placeholders['center_x_res_labels'] - center_x_residuals_normalized, axis=-1)
        center_x_res_loss = huber_loss(train_reg_mask*center_x_residuals_dist, delta=1.0)
        bin_z_onehot = tf.one_hot(self.placeholders['center_bin_z_labels'],
            depth=NUM_CENTER_BIN,
            on_value=1, off_value=0, axis=-1) # BxNUM_CENTER_BIN
        center_z_residuals_normalized = tf.reduce_sum(end_points['center_z_residuals_normalized']*tf.to_float(bin_z_onehot), axis=-1) # B
        center_z_residuals_dist = tf.norm(self.placeholders['center_z_res_labels'] - center_z_residuals_normalized, axis=-1)
        center_z_res_loss = huber_loss(train_reg_mask*center_z_residuals_dist, delta=1.0)
        # y is directly regressed
        center_y_residuals_dist = tf.norm(self.placeholders['center_y_res_labels'] - tf.gather(end_points['center_y_residuals'], 0, axis=-1), axis=-1)
        center_y_res_loss = huber_loss(train_reg_mask*center_y_residuals_dist, delta=1.0)
        tf.summary.scalar('center_x  class loss', center_x_cls_loss)
        tf.summary.scalar('center_z  class loss', center_z_cls_loss)
        tf.summary.scalar('center_x residual loss', center_x_res_loss)
        tf.summary.scalar('center_y residual loss', center_y_res_loss)
        tf.summary.scalar('center_z residual loss', center_z_res_loss)
        # Heading loss
        heading_class_loss = tf.reduce_mean( \
            train_reg_mask*tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['heading_scores'], labels=self.placeholders['heading_bin_labels']))
        hcls_onehot = tf.one_hot(self.placeholders['heading_bin_labels'],
            depth=NUM_HEADING_BIN,
            on_value=1, off_value=0, axis=-1) # BxNxNUM_HEADING_BIN
        heading_residual_normalized_label = self.placeholders['heading_res_labels']
        heading_res_dist = tf.norm(tf.reduce_sum( \
            end_points['heading_residuals_normalized']*tf.to_float(hcls_onehot), axis=-1) - \
            heading_residual_normalized_label)
        heading_res_loss = huber_loss(train_reg_mask*heading_res_dist, delta=1.0)
        tf.summary.scalar('heading class loss', heading_class_loss)
        tf.summary.scalar('heading residual loss', heading_res_loss)
        # Size loss
        size_class_loss = tf.reduce_mean( \
            train_reg_mask*tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['size_scores'], labels=self.placeholders['size_class_labels']))

        scls_onehot = tf.one_hot(self.placeholders['size_class_labels'],
            depth=NUM_SIZE_CLUSTER,
            on_value=1, off_value=0, axis=-1) # BxNUM_SIZE_CLUSTER
        scls_onehot_tiled = tf.tile(tf.expand_dims( \
            tf.to_float(scls_onehot), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
        predicted_size_residual_normalized = tf.reduce_sum( \
            end_points['size_residuals_normalized']*scls_onehot_tiled, axis=1) # Bx3

        size_residual_label_normalized = self.placeholders['size_res_labels'] # Bx3

        size_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
        size_res_loss = huber_loss(train_reg_mask*size_dist, delta=1.0)
        tf.summary.scalar('size class loss', size_class_loss)
        tf.summary.scalar('size residual loss', size_res_loss)

        obj_cls_weight = 1
        cls_weight = 1
        res_weight = 1
        total_loss = obj_cls_weight * cls_loss + \
            cls_weight * (center_x_cls_loss + center_z_cls_loss + heading_class_loss + size_class_loss) + \
            res_weight * (0.1*center_x_res_loss + 0.1*center_z_res_loss + center_y_res_loss + 0.1*heading_res_loss + size_res_loss)

        loss_endpoints = {
            #'size_class_loss': size_class_loss,
            'size_res_loss': size_res_loss,
            #'heading_class_loss': heading_class_loss,
            #'heading_res_loss': heading_res_loss,
            #'center_x_cls_loss': center_x_cls_loss,
            #'center_z_cls_loss': center_z_cls_loss,
            #'center_x_res_loss': center_x_res_loss,
            #'center_z_res_loss': center_z_res_loss,
            #'center_y_res_loss': center_y_res_loss,
            #'mask_loss': cls_loss
            #'mean_size_label': mean_size_label,
            'size_residuals_normalized': end_points['size_residuals_normalized']
        }
        return total_loss, loss_endpoints

if __name__ == '__main__':
    with tf.Graph().as_default():
        model = RCNN(32, 512)
        for key in model.end_points:
            print((key, model.end_points[key]))
