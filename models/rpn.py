from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from model_util import point_cloud_masking
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss
from model_util import NUM_CHANNEL, NUM_FG_POINT
from parameterize import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CENTER_BIN

def get_segmentation_net(point_cloud, is_training, bn_decay, end_points):
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
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,NUM_CHANNEL-3])

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points,
        128, [0.2,0.4,0.8], [32,64,128],
        [[32,32,64], [64,64,128], [64,96,128]],
        is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points,
        32, [0.4,0.8,1.6], [64,64,128],
        [[64,64,128], [128,128,256], [128,128,256]],
        is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[128,256,1024],
        mlp2=None, group_all=True, is_training=is_training,
        bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers

    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
        [128,128], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
        [128,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
        tf.concat([l0_xyz,l0_points],axis=-1), l1_points,
        [128,128], is_training, bn_decay, scope='fa_layer3')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
        is_training=is_training, scope='conv1d-fc1', bn_decay=bn_decay)
    end_points['point_feats'] = net
    net = tf_util.dropout(net, keep_prob=0.7,
        is_training=is_training, scope='dp1')
    logits = tf_util.conv1d(net, 2, 1,
        padding='VALID', activation_fn=None, scope='conv1d-fc2')
    end_points['foreground_logits'] = logits

    return end_points

def get_region_proposal_net(point_feats, is_training, bn_decay, end_points):
    batch_size = point_feats.get_shape()[0].value
    npoints = point_feats.get_shape()[1].value
    point_feats = tf.slice(point_feats, [0,0,3], [-1,-1,-1]) # (B, N, D)
    net = tf.reshape(point_feats, [batch_size * npoints, -1])
    # Fully connected layers
    net = tf_util.fully_connected(net, 512, bn=True,
        is_training=is_training, scope='rp-fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True,
        is_training=is_training, scope='rp-fc2', bn_decay=bn_decay)

    # The first 2 numbers: box objectness logits,
    # the next NUM_CENTER_BIN*2*2: CENTER_BIN class scores and bin residuals for (x,z)
    # next 1: center residual for y
    # next NUM_HEADING_BIN*2: heading bin class scores and residuals
    # next NUM_SIZE_CLUSTER*4: size cluster class scores and residuals(l,w,h)
    output = tf_util.fully_connected(net,
        2+NUM_CENTER_BIN*2*2+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4,
        activation_fn=None, scope='rp-fc3')
    end_points['proposals'] = output
    return output

def get_model(point_cloud, mask_label, is_training, bn_decay, end_points):
    end_points = get_segmentation_net(point_cloud, is_training, bn_decay, end_points)
    foreground_logits = tf.cond(is_training, lambda: tf.one_hot(mask_label, 2), lambda: end_points['foreground_logits'])
    fg_point_feats, end_points = point_cloud_masking(
        end_points['point_feats'], foreground_logits,
        end_points, xyz_only=False) # BxNUM_FG_POINTxD
    proposals = get_region_proposal_net(fg_point_feats, is_training, bn_decay, end_points)
    batch_size = fg_point_feats.get_shape()[0].value
    proposals_reshaped = tf.reshape(proposals, [batch_size, NUM_FG_POINT, -1])
    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(proposals_reshaped, end_points)
    return end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,4))
        outputs = get_model(inputs, tf.constant(True), None, {})
        for key in outputs:
            print((key, outputs[key]))
        # loss = get_loss(tf.zeros((32,),dtype=tf.int32),
        #     tf.zeros((32,1024),dtype=tf.int32),
        #     tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32),
        #     tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32),
        #     tf.zeros((32,3)), outputs)
        # print(loss)
