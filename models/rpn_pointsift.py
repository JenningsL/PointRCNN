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
from rpn import RPN, NUM_SEG_CLASSES, NUM_CENTER_BIN, NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from pointSIFT_util import pointSIFT_module, pointSIFT_res_module, pointnet_fp_module, pointnet_sa_module

class RPN_PointSIFT(RPN):
    """docstring for RPN."""
    def __init__(self, batch_size, num_point, num_channel=4, bn_decay=None, is_training=True):
        super(RPN_PointSIFT, self).__init__(batch_size, num_point, num_channel, bn_decay, is_training)

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
        c0_l0_xyz, c0_l0_points, c0_l0_indices = pointSIFT_res_module(l0_xyz, l0_points, radius=0.1, out_channel=16, is_training=is_training, bn_decay=bn_decay, scope='layer0_c0', merge='concat')

        l1_xyz, l1_points = pointnet_sa_module_msg(c0_l0_xyz, c0_l0_points,
            4096, [0.1,0.5], [16,32],
            [[16,32], [32,32]],
            is_training, bn_decay, scope='layer1', bn=True)
        c0_l1_xyz, c0_l1_points, c0_l1_indices = pointSIFT_res_module(l1_xyz, l1_points, radius=0.5, out_channel=64, is_training=is_training, bn_decay=bn_decay, scope='layer1_c0')

        l2_xyz, l2_points = pointnet_sa_module_msg(c0_l1_xyz, c0_l1_points,
            1024, [0.5,1.0], [16,32],
            [[64,128], [96,128]],
            is_training, bn_decay, scope='layer2', bn=True)
        c0_l2_xyz, c0_l2_points, c0_l2_indices = pointSIFT_res_module(l2_xyz, l2_points, radius=1.0, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='layer2_c0')

        l3_xyz, l3_points = pointnet_sa_module_msg(c0_l2_xyz, c0_l2_points,
            256, [1.0,2.0], [16,32],
            [[196, 256], [196, 256]],
            is_training, bn_decay, scope='layer3', bn=True)
        c0_l3_xyz, c0_l3_points, c0_l3_indices = pointSIFT_res_module(l3_xyz, l3_points, radius=2.0, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='layer3_c0')

        l4_xyz, l4_points = pointnet_sa_module_msg(c0_l3_xyz, c0_l3_points,
            64, [2.0,4.0], [16,32],
            [[256, 512], [384, 512]],
            is_training, bn_decay, scope='layer4', bn=True)

        # Feature Propagation layers
        l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points,
            [512], is_training, bn_decay, scope='fa_layer2', bn=True)
        #_, l3_points_1, _ = pointSIFT_module(l3_xyz, l3_points, radius=4.0, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c0')
        #_, l3_points_2, _ = pointSIFT_module(l3_xyz, l3_points, radius=4.0, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c1')
        #l3_points = tf.concat([l3_points_1, l3_points_2], axis=-1)
        #l3_points = tf_util.conv1d(l3_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_2_fc', bn_decay=bn_decay)
        _, l3_points, _ = pointSIFT_module(l3_xyz, l3_points, radius=4.0, out_channel=512, is_training=is_training, bn_decay=bn_decay, scope='fa_layer2_c0')

        l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
            [256], is_training, bn_decay, scope='fa_layer3', bn=True)
        #_, l2_points_1, _ = pointSIFT_module(l2_xyz, l2_points, radius=2.0, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c0')
        #_, l2_points_2, _ = pointSIFT_module(l2_xyz, l2_points, radius=2.0, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c1')
        #l2_points = tf.concat([l2_points_1, l2_points_2], axis=-1)
        #l2_points = tf_util.conv1d(l2_points, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fa_3_fc', bn_decay=bn_decay)
        _, l2_points, _ = pointSIFT_module(l2_xyz, l2_points, radius=2.0, out_channel=256, is_training=is_training, bn_decay=bn_decay, scope='fa_layer3_c1')

        l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
            [128], is_training, bn_decay, scope='fa_layer4', bn=True)
        _, l1_points, _ = pointSIFT_module(l1_xyz, l1_points, radius=1.0, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='fa_layer4_c0')

        l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
            l0_points, l1_points,
            [128], is_training, bn_decay, scope='fa_layer5', bn=True)
        _, l0_points, _ = pointSIFT_module(l0_xyz, l0_points, radius=0.5, out_channel=128, is_training=is_training, bn_decay=bn_decay, scope='fa_layer5_c0')

        end_points['point_feats'] = tf.concat([l0_xyz,l0_points], axis=-1) # (B, N, 3+C1)
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

    def get_region_proposal_net(self, point_feats, is_training, bn_decay, end_points):
        batch_size = point_feats.get_shape()[0].value
        npoints = point_feats.get_shape()[1].value
        # xyz is not used
        point_feats = tf.slice(point_feats, [0,0,3], [-1,-1,-1]) # (B, N, D)
        # FC layers
        net = tf_util.conv1d(point_feats, 128, 1, padding='VALID', bn=True,
            is_training=is_training, scope='rp-conv1d-fc1', bn_decay=bn_decay)
        #net = tf_util.dropout(net, keep_prob=0.5,
        #    is_training=is_training, scope='rp-dp1')
        #net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True,
        #    is_training=is_training, scope='rp-conv1d-fc2', bn_decay=bn_decay)
        # The first NUM_CENTER_BIN*2*2: CENTER_BIN class scores and bin residuals for (x,z)
        # next 1: center residual for y
        # next NUM_HEADING_BIN*2: heading bin class scores and residuals
        # next NUM_SIZE_CLUSTER*4: size cluster class scores and residuals(l,w,h)
        output = tf_util.conv1d(net, NUM_CENTER_BIN*2*2+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, 1,
            padding='VALID', activation_fn=None, scope='rp-conv1d-fc-out')
        end_points['proposals'] = output
        return output

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
