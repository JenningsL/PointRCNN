from __future__ import print_function

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
from model_util import point_cloud_masking, get_box3d_corners_helper, focal_loss, huber_loss, SigmoidFocalClassificationLoss
from model_util import NUM_FG_POINT
from box_encoder import NUM_SIZE_CLUSTER, type_mean_size
from box_encoder import BoxEncoder
import projection
from img_vgg_pyramid import ImgVggPyr
from collections import namedtuple

NUM_HEADING_BIN = 12
NUM_CENTER_BIN = 12
CENTER_SEARCH_RANGE = 3.0
HEADING_SEARCH_RANGE = np.pi
NUM_SEG_CLASSES = 2

class RPN(object):
    """docstring for RPN."""
    def __init__(self, batch_size, num_point, num_channel=4, bn_decay=None, is_training=True):
        self.batch_size = batch_size
        self.num_point = num_point
        self.num_channel = num_channel
        self.bn_decay = bn_decay
        self.is_training = is_training
        self.end_points = {}
        self.box_encoder = BoxEncoder(CENTER_SEARCH_RANGE, NUM_CENTER_BIN, HEADING_SEARCH_RANGE, NUM_HEADING_BIN)
        self.placeholders = self.get_placeholders()
        self.build()

    def get_placeholders(self):
        batch_size = self.batch_size
        num_point = self.num_point
        return {
            'pointclouds': tf.placeholder(tf.float32, shape=(batch_size, num_point, self.num_channel)),
            'img_inputs': tf.placeholder(tf.float32, shape=(batch_size, 360, 1200, 3)),
            'calib': tf.placeholder(tf.float32, shape=(batch_size, 3, 4)),
            'seg_labels': tf.placeholder(tf.int32, shape=(batch_size, num_point)),
            'center_bin_x_labels': tf.placeholder(tf.int32, shape=(batch_size, num_point)),
            'center_bin_z_labels': tf.placeholder(tf.int32, shape=(batch_size, num_point)),
            'center_x_residuals_labels': tf.placeholder(tf.float32, shape=(batch_size, num_point)),
            'center_z_residuals_labels': tf.placeholder(tf.float32, shape=(batch_size, num_point)),
            'center_y_residuals_labels': tf.placeholder(tf.float32, shape=(batch_size, num_point)),
            'heading_bin_labels': tf.placeholder(tf.int32, shape=(batch_size, num_point)),
            'heading_residuals_labels': tf.placeholder(tf.float32, shape=(batch_size, num_point)),
            'size_class_labels': tf.placeholder(tf.int32, shape=(batch_size, num_point)),
            'size_residuals_labels': tf.placeholder(tf.float32, shape=(batch_size, num_point, 3)),
            'gt_boxes': tf.placeholder(tf.float32, shape=(batch_size, None, 8, 3)),
            'gt_box_of_point': tf.placeholder(tf.float32, shape=(batch_size, num_point, 8, 3)),
            'img_seg_softmax': tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_SEG_CLASSES)),
            'is_training_pl': tf.placeholder(tf.bool, shape=())
        }

    def parse_output_to_tensors(self, output, end_points):
        ''' Parse batch output to separate tensors (added to end_points)
        Input:
            output: TF tensor in shape (B,N,NUM_CENTER_BIN*2*2+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
            end_points: dict
        Output:
            end_points: dict (updated)
        '''
        batch_size = output.get_shape()[0].value
        npoints = output.get_shape()[1].value
        # objectness and center
        #end_points['objectness'] = tf.slice(output, [0,0,0], [-1,-1,2])
        center_x_scores = tf.slice(output, [0,0,0], [-1,-1,NUM_CENTER_BIN])
        center_x_residuals_normalized = tf.slice(output, [0,0,NUM_CENTER_BIN],
            [-1,-1,NUM_CENTER_BIN])
        end_points['center_x_scores'] = center_x_scores # (B,N,NUM_CENTER_BIN)
        end_points['center_x_residuals_normalized'] = \
            center_x_residuals_normalized # (B,N,NUM_CENTER_BIN)
        center_z_scores = tf.slice(output, [0,0,NUM_CENTER_BIN*2], [-1,-1,NUM_CENTER_BIN])
        center_z_residuals_normalized = tf.slice(output, [0,0,NUM_CENTER_BIN*3],
            [-1,-1,NUM_CENTER_BIN])
        end_points['center_z_scores'] = center_z_scores # (B,N,NUM_CENTER_BIN)
        end_points['center_z_residuals_normalized'] = \
            center_z_residuals_normalized # (B,N,NUM_CENTER_BIN)
        end_points['center_y_residuals'] = tf.slice(output, [0,0,NUM_CENTER_BIN*4], [-1,-1,1])
        # heading
        heading_scores = tf.slice(output, [0,0,NUM_CENTER_BIN*4+1], [-1,-1,NUM_HEADING_BIN])
        heading_residuals_normalized = tf.slice(output, [0,0,NUM_CENTER_BIN*4+1+NUM_HEADING_BIN],
            [-1,-1,NUM_HEADING_BIN])
        end_points['heading_scores'] = heading_scores # (B,N,NUM_HEADING_BIN)
        end_points['heading_residuals_normalized'] = heading_residuals_normalized # (B,N,NUM_HEADING_BIN)
        # end_points['heading_residuals'] = \
        #     heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN
        # size
        size_scores = tf.slice(output, [0,0,NUM_CENTER_BIN*4+1+NUM_HEADING_BIN*2],
            [-1,-1,NUM_SIZE_CLUSTER]) # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = tf.slice(output,
            [0,0,NUM_CENTER_BIN*4+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER], [-1,-1,NUM_SIZE_CLUSTER*3])
        size_residuals_normalized = tf.reshape(size_residuals_normalized,
            [batch_size, npoints, NUM_SIZE_CLUSTER, 3])
        end_points['size_scores'] = size_scores
        end_points['size_residuals_normalized'] = size_residuals_normalized
        # end_points['size_residuals'] = size_residuals_normalized * \
        #     tf.expand_dims(tf.constant(type_mean_size, dtype=tf.float32), 0)
        box_center, box_angle, box_size = self.box_encoder.tf_decode(end_points)
        box_center = box_center + end_points['fg_points_xyz']
        box_num = batch_size * npoints
        corners_3d = get_box3d_corners_helper(
            tf.reshape(box_center, [box_num,3]), tf.reshape(box_angle, [box_num]), tf.reshape(box_size, [box_num,3]))
        end_points['proposal_boxes'] = tf.reshape(corners_3d, [batch_size, npoints, 8, 3])
        return end_points

    def build_img_extractor(self):
        self._img_pixel_size = np.asarray([360, 1200])
        VGG_config = namedtuple('VGG_config', 'vgg_conv1 vgg_conv2 vgg_conv3 vgg_conv4 l2_weight_decay')
        self._img_feature_extractor = ImgVggPyr(VGG_config(**{
            'vgg_conv1': [2, 32],
            'vgg_conv2': [2, 64],
            'vgg_conv3': [3, 128],
            'vgg_conv4': [3, 256],
            'l2_weight_decay': 0.0005
        }))
        self._img_preprocessed = \
            self._img_feature_extractor.preprocess_input(
                self.placeholders['img_inputs'], self._img_pixel_size)
        self.img_feature_maps, self.img_end_points = \
            self._img_feature_extractor.build(
                self._img_preprocessed,
                self._img_pixel_size,
                self.is_training)
        #return self.img_feature_maps
        self.img_bottleneck = slim.conv2d(
            self.img_feature_maps,
            #128, [1, 1],
            2, [1, 1],
            scope='bottleneck',
            normalizer_fn=slim.batch_norm,
            #normalizer_fn=None,
            normalizer_params={
                'is_training': self.is_training})
        return self.img_bottleneck

    def get_segmentation_net(self, point_cloud, is_training, bn_decay, end_points):
        ''' 3D instance segmentation PointNet v2 network.
        Input:
            point_cloud: TF tensor in shape (B,N,4)
                frustum point clouds with XYZ and intensity in point channels
                XYZs are in frustum coordinate
            is_training: TF boolean scalar
            bn_decay: TF float scalar
            end_points: dict
        Output:
            logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
            end_points: dict
        '''
        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        if self.num_channel > 3:
            l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,self.num_channel-3])
        else:
            l0_points = None

        # Set abstraction layers
        l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points,
            4096, [0.1,0.5], [16,32],
            [[16,16,32], [32,32,64]],
            is_training, bn_decay, scope='layer1', bn=True)
        l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points,
            1024, [0.5,1.0], [16,32],
            [[64,64,128], [64,96,128]],
            is_training, bn_decay, scope='layer2', bn=True)
        l3_xyz, l3_points = pointnet_sa_module_msg(l2_xyz, l2_points,
            256, [1.0,2.0], [16,32],
            [[128, 196, 256], [128,196, 256]],
            is_training, bn_decay, scope='layer3', bn=True)
        l4_xyz, l4_points = pointnet_sa_module_msg(l3_xyz, l3_points,
            64, [2.0,4.0], [16,32],
            [[256, 256, 512], [256,384, 512]],
            is_training, bn_decay, scope='layer4', bn=True)

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points,
            [512,512], is_training, bn_decay, scope='fa_layer2', bn=True)
        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
            [512,512], is_training, bn_decay, scope='fa_layer3', bn=True)
        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
            [256,256], is_training, bn_decay, scope='fa_layer4', bn=True)
        l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
            tf.concat([l0_xyz,l0_points],axis=-1), l1_points,
            [128,128], is_training, bn_decay, scope='fa_layer5', bn=True)
        end_points['point_feats'] = tf.concat([l0_xyz,l0_points], axis=-1) # (B, N, 3+C1)
        #end_points['point_feats_fuse'] = tf.concat([end_points['point_feats'], end_points['point_img_feats']], axis=-1) # (B, N, 3+C1+C2)
        #semantic_features = tf.concat([l0_points, end_points['point_img_feats']], axis=-1) # (B, N, C1+C2)
        #end_points['point_feats_fuse'] = end_points['point_feats']
        semantic_features = l0_points
        # FC layers
        net = tf_util.conv1d(semantic_features, 128, 1, padding='VALID', bn=True,
            is_training=is_training, scope='conv1d-fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5,
            is_training=is_training, scope='dp1')
        logits = tf_util.conv1d(net, NUM_SEG_CLASSES, 1,
            padding='VALID', activation_fn=None, scope='conv1d-fc2')
        end_points['foreground_logits'] = logits

        return end_points

    def reduce_proposals(self, end_points):
        '''Use NMS to reduce the number of proposals'''
        batch_size = end_points['fg_points_xyz'].shape[0]
        # confidence
        fg_logits = tf.gather_nd(end_points['foreground_logits'], end_points['fg_point_indices']) # (B,M)
        seg_scores = tf.reduce_max(tf.nn.softmax(fg_logits), axis=-1) # (B,M)
        bin_x_scores = tf.reduce_max(tf.nn.softmax(end_points['center_x_scores']), axis=-1) # (B,M)
        bin_z_scores = tf.reduce_max(tf.nn.softmax(end_points['center_z_scores']), axis=-1) # (B,M)
        heading_scores = tf.reduce_max(tf.nn.softmax(end_points['heading_scores']), axis=-1) # (B,M)
        size_scores = tf.reduce_max(tf.nn.softmax(end_points['size_scores']), axis=-1) # (B,M)
        # confidence = seg_scores + bin_x_scores + bin_z_scores + heading_scores + size_scores
        confidence = seg_scores * bin_x_scores * bin_z_scores * heading_scores * size_scores
        confidence.set_shape([batch_size, NUM_FG_POINT])
        end_points['proposal_scores'] = confidence
        # BEV boxes
        boxes_3d = end_points['proposal_boxes'] # (B,M,8,3)
        corners_min = tf.gather(tf.reduce_min(boxes_3d, axis=2), [0,2], axis=-1)
        corners_max = tf.gather(tf.reduce_max(boxes_3d, axis=2), [0,2], axis=-1) # (B,M,2) x,z
        boxes_bev = tf.concat([corners_min, corners_max], axis=-1) # (B,M,4)
        boxes_bev.set_shape([batch_size, NUM_FG_POINT,4])

        confidence_unpack = tf.unstack(confidence, axis=0)
        boxes_bev_unpack = tf.unstack(boxes_bev, axis=0)
        #boxes_3d_unpack = tf.unstack(end_points['proposal_boxes'], axis=0)
        #boxes_3d_list = []
        batch_nms_indices = []
        for i in range(len(confidence_unpack)):
            nms_indices = tf.image.non_max_suppression(boxes_bev_unpack[i], confidence_unpack[i], 300) # at most 300
            #boxes_3d_list.append(tf.gather(boxes_3d_unpack[i], nms_indices))
            nms_indices = tf.pad(nms_indices, [[0, NUM_FG_POINT-tf.shape(nms_indices)[0]]], mode='CONSTANT', constant_values=-1)
            batch_nms_indices.append(nms_indices)
        end_points['nms_indices'] = tf.stack(batch_nms_indices, axis=0)
        return end_points

    def get_region_proposal_net(self, point_feats, is_training, bn_decay, end_points):
        batch_size = point_feats.get_shape()[0].value
        npoints = point_feats.get_shape()[1].value
        # xyz is not used
        point_feats = tf.slice(point_feats, [0,0,3], [-1,-1,-1]) # (B, N, D)
        # FC layers
        net = tf_util.conv1d(point_feats, 256, 1, padding='VALID', bn=True,
            is_training=is_training, scope='rp-conv1d-fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5,
            is_training=is_training, scope='rp-dp1')
        #net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True,
        #    is_training=is_training, scope='rp-conv1d-fc2', bn_decay=bn_decay)
        #net = tf_util.dropout(net, keep_prob=0.5,
        #    is_training=is_training, scope='rp-dp2')
        output = tf_util.conv1d(net, NUM_CENTER_BIN*2*2+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, 1,
            padding='VALID', activation_fn=None, scope='rp-conv1d-fc-out')
        end_points['proposals'] = output
        return output

    def build(self):
        point_cloud = self.placeholders['pointclouds']
        is_training = self.placeholders['is_training_pl']
        mask_label = self.placeholders['seg_labels']
        bn_decay = self.bn_decay
        end_points = self.end_points

        end_points = self.get_segmentation_net(point_cloud, is_training, bn_decay, end_points)
        seg_softmax = tf.nn.softmax(end_points['foreground_logits'], axis=-1) + self.placeholders['img_seg_softmax']
        seg_logits = tf.cond(is_training, lambda: tf.one_hot(mask_label, NUM_SEG_CLASSES), lambda: seg_softmax)
        #end_points['point_feats_fuse'] = tf.concat([end_points['point_feats_fuse'], seg_logits], axis=-1)
        # fg_point_feats include xyz
        fg_point_feats, end_points = point_cloud_masking(
            end_points['point_feats'], seg_logits,
            end_points, xyz_only=False) # BxNUM_FG_POINTxD
        proposals = self.get_region_proposal_net(fg_point_feats, is_training, bn_decay, end_points)
        proposals_reshaped = tf.reshape(proposals, [self.batch_size, NUM_FG_POINT, -1])
        # Parse output to 3D box parameters
        end_points = self.parse_output_to_tensors(proposals_reshaped, end_points)
        end_points = self.reduce_proposals(end_points)
        # for iou eval
        end_points['gt_box_of_point'] = tf.gather_nd(self.placeholders['gt_box_of_point'], end_points['fg_point_indices'])
        end_points['gt_box_of_point'].set_shape([self.batch_size, NUM_FG_POINT, 8, 3])
        return end_points

    def get_loss(self):
        pls = self.placeholders
        end_points = self.end_points
        batch_size = self.batch_size
        num_point = self.num_point
        # 3D Segmentation loss
        #mask_loss = focal_loss(end_points['foreground_logits'], tf.one_hot(pls['seg_labels'], NUM_SEG_CLASSES, axis=-1))
        focal_loss = SigmoidFocalClassificationLoss()
        mask_weights = tf.tile(tf.constant([[[1, 15]]], dtype=tf.float32), [batch_size, num_point, 1])
        pos_normalizer = tf.maximum(tf.reduce_sum(tf.cast(pls['seg_labels']>0, tf.float32)), 1)
        mask_weights = mask_weights / pos_normalizer
        mask_loss = focal_loss._compute_loss(end_points['foreground_logits'], tf.one_hot(pls['seg_labels'], NUM_SEG_CLASSES, axis=-1), mask_weights)
        tf.summary.scalar('mask loss', mask_loss)
        #return mask_loss, {}
        # gather box estimation labels of foreground points
        labels_fg = {}
        for k in pls.keys():
            if k not in ['center_bin_x_labels','center_bin_z_labels','center_x_residuals_labels',
                'center_z_residuals_labels','center_y_residuals_labels','heading_bin_labels',
                'heading_residuals_labels','size_class_labels','size_residuals_labels',]:
                continue
            labels_fg[k] = tf.gather_nd(pls[k], end_points['fg_point_indices'])
            if k == 'size_residuals_labels':
                labels_fg[k].set_shape([batch_size, NUM_FG_POINT, 3])
            else:
                labels_fg[k].set_shape([batch_size, NUM_FG_POINT])
        # Center loss
        center_x_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
           logits=end_points['center_x_scores'], labels=labels_fg['center_bin_x_labels']))
        center_z_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
           logits=end_points['center_z_scores'], labels=labels_fg['center_bin_z_labels']))
        bin_x_onehot = tf.one_hot(labels_fg['center_bin_x_labels'],
            depth=NUM_CENTER_BIN,
            on_value=1, off_value=0, axis=-1) # BxNxNUM_CENTER_BIN
        # NOTICE: labels['center_x_residuals'] is already normalized
        center_x_residuals_normalized = tf.reduce_sum(end_points['center_x_residuals_normalized']*tf.to_float(bin_x_onehot), axis=2) # BxN
        center_x_residuals_dist = tf.norm(labels_fg['center_x_residuals_labels'] - center_x_residuals_normalized, axis=-1)
        center_x_res_loss = huber_loss(center_x_residuals_dist, delta=1.0)
        bin_z_onehot = tf.one_hot(labels_fg['center_bin_z_labels'],
            depth=NUM_CENTER_BIN,
            on_value=1, off_value=0, axis=-1) # BxNxNUM_CENTER_BIN
        center_z_residuals_normalized = tf.reduce_sum(end_points['center_z_residuals_normalized']*tf.to_float(bin_z_onehot), axis=2) # BxN
        center_z_residuals_dist = tf.norm(labels_fg['center_z_residuals_labels'] - center_z_residuals_normalized, axis=-1)
        center_z_res_loss = huber_loss(center_z_residuals_dist, delta=1.0)
        # y is directly regressed
        center_y_residuals_dist = tf.norm(labels_fg['center_y_residuals_labels'] - tf.gather(end_points['center_y_residuals'], 0, axis=-1), axis=-1)
        center_y_res_loss = huber_loss(center_y_residuals_dist, delta=1.0)
        tf.summary.scalar('center_x  class loss', center_x_cls_loss)
        tf.summary.scalar('center_z  class loss', center_z_cls_loss)
        tf.summary.scalar('center_x residual loss', center_x_res_loss)
        tf.summary.scalar('center_y residual loss', center_y_res_loss)
        tf.summary.scalar('center_z residual loss', center_z_res_loss)
        # Heading loss
        heading_class_loss = tf.reduce_mean( \
            tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['heading_scores'], labels=labels_fg['heading_bin_labels']))
        hcls_onehot = tf.one_hot(labels_fg['heading_bin_labels'],
            depth=NUM_HEADING_BIN,
            on_value=1, off_value=0, axis=-1) # BxNxNUM_HEADING_BIN
        heading_residual_normalized_label = labels_fg['heading_residuals_labels']
        heading_res_dist = tf.norm(heading_residual_normalized_label - tf.reduce_sum( \
            end_points['heading_residuals_normalized']*tf.to_float(hcls_onehot), axis=2))
        heading_res_loss = huber_loss(heading_res_dist, delta=1.0)
        tf.summary.scalar('heading class loss', heading_class_loss)
        tf.summary.scalar('heading residual loss', heading_res_loss)
        # Size loss
        size_class_loss = tf.reduce_mean( \
            tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['size_scores'], labels=labels_fg['size_class_labels']))

        scls_onehot = tf.one_hot(labels_fg['size_class_labels'],
            depth=NUM_SIZE_CLUSTER,
            on_value=1, off_value=0, axis=-1) # BxNxNUM_SIZE_CLUSTER
        scls_onehot_tiled = tf.tile(tf.expand_dims( \
            tf.to_float(scls_onehot), -1), [1,1,1,3]) # BxNxNUM_SIZE_CLUSTERx3
        predicted_size_residual_normalized = tf.reduce_sum( \
            end_points['size_residuals_normalized']*scls_onehot_tiled, axis=2) # BxNx3

        size_residual_label_normalized = labels_fg['size_residuals_labels'] # BxNx3

        size_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
        size_res_loss = huber_loss(size_dist, delta=1.0)
        tf.summary.scalar('size class loss', size_class_loss)
        tf.summary.scalar('size residual loss', size_res_loss)

        seg_weight = 1
        cls_weight = 1
        res_weight = 1
        total_loss = seg_weight * mask_loss + \
            cls_weight * (center_x_cls_loss + center_z_cls_loss + heading_class_loss + size_class_loss) + \
            res_weight * (0.1*center_x_res_loss + 0.1*center_z_res_loss + 0.1*center_y_res_loss + 0.1*heading_res_loss + size_res_loss)
        loss_endpoints = {
            'size_class_loss': size_class_loss,
            'size_res_loss': size_res_loss,
            'heading_class_loss': heading_class_loss,
            'heading_res_loss': heading_res_loss,
            'center_x_cls_loss': center_x_cls_loss,
            'center_z_cls_loss': center_z_cls_loss,
            'center_x_res_loss': center_x_res_loss,
            'center_z_res_loss': center_z_res_loss,
            'center_y_res_loss': center_y_res_loss,
            'mask_loss': mask_loss
        }

        return total_loss, loss_endpoints

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,16384,4))
        mask_label = tf.zeros((32,16384), dtype=tf.int32)
        gt_box_of_point = tf.zeros((32,16384,8,3), dtype=tf.float32)
        labels_pl = {'mask_label': mask_label, 'gt_box_of_point': gt_box_of_point}
        outputs = get_model(inputs, labels_pl, tf.constant(True), None, {})
        for key in outputs:
            print((key, outputs[key]))
        # loss = get_loss(tf.zeros((32,),dtype=tf.int32),
        #     tf.zeros((32,1024),dtype=tf.int32),
        #     tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32),
        #     tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32),
        #     tf.zeros((32,3)), outputs)
        # print(loss)
