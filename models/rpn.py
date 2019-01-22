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
    '''
    l5_xyz, l5_points, _ = pointnet_sa_module(l4_xyz, l4_points,
        npoint=None, radius=None, nsample=None, mlp=[128,256,1024],
        mlp2=None, group_all=True, is_training=is_training,
        bn_decay=bn_decay, scope='layer5')
    '''

    # Feature Propagation layers
    '''
    l4_points = pointnet_fp_module(l4_xyz, l5_xyz, l4_points, l5_points,
        [128,128], is_training, bn_decay, scope='fa_layer1')
    '''
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points,
        [512,512], is_training, bn_decay, scope='fa_layer2', bn=True)
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points,
        [512,512], is_training, bn_decay, scope='fa_layer3', bn=True)
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points,
        [256,256], is_training, bn_decay, scope='fa_layer4', bn=True)
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz,
        tf.concat([l0_xyz,l0_points],axis=-1), l1_points,
        [128,128], is_training, bn_decay, scope='fa_layer5', bn=True)
    end_points['point_feats'] = tf.concat([l0_xyz,l0_points], axis=-1) # (B, N, 3+C)
    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
        is_training=is_training, scope='conv1d-fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7,
        is_training=is_training, scope='dp1')
    logits = tf_util.conv1d(net, 2, 1,
        padding='VALID', activation_fn=None, scope='conv1d-fc2')
    end_points['foreground_logits'] = logits

    return end_points

def reduce_proposals(end_points):
    '''Use NMS to reduce the number of proposals'''
    batch_size = end_points['fg_points_xyz'].shape[0]
    # confidence
    fg_logits = tf.gather_nd(end_points['foreground_logits'], end_points['fg_point_indices']) # (B,M)
    seg_scores = tf.reduce_max(tf.nn.softmax(fg_logits), axis=-1) # (B,M)
    bin_x_scores = tf.reduce_max(tf.nn.softmax(end_points['center_x_scores']), axis=-1) # (B,M)
    bin_z_scores = tf.reduce_max(tf.nn.softmax(end_points['center_z_scores']), axis=-1) # (B,M)
    heading_scores = tf.reduce_max(tf.nn.softmax(end_points['heading_scores']), axis=-1) # (B,M)
    size_scores = tf.reduce_max(tf.nn.softmax(end_points['size_scores']), axis=-1) # (B,M)
    confidence = seg_scores + bin_x_scores + bin_z_scores + heading_scores + size_scores
    confidence.set_shape([batch_size, NUM_FG_POINT])
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
        nms_indices = tf.image.non_max_suppression(boxes_bev_unpack[i], confidence_unpack[i], 300)
        #boxes_3d_list.append(tf.gather(boxes_3d_unpack[i], nms_indices))
        nms_indices = tf.pad(nms_indices, [[0, 300-tf.shape(nms_indices)[0]]], mode='CONSTANT', constant_values=-1)
        batch_nms_indices.append(nms_indices)
    end_points['nms_indices'] = tf.stack(batch_nms_indices, axis=0)
    return end_points

def get_region_proposal_net(point_feats, is_training, bn_decay, end_points):
    batch_size = point_feats.get_shape()[0].value
    npoints = point_feats.get_shape()[1].value
    point_feats = tf.slice(point_feats, [0,0,3], [-1,-1,-1]) # (B, N, D)
    net = tf.reshape(point_feats, [batch_size * npoints, -1])
    # Fully connected layers
    net = tf_util.fully_connected(net, 256, bn=True,
        is_training=is_training, scope='rp-fc0', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7,
    #    is_training=is_training, scope='rp-dp0')
    net = tf_util.fully_connected(net, 256, bn=True,
        is_training=is_training, scope='rp-fc1', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7,
    #    is_training=is_training, scope='rp-dp1')
    net = tf_util.fully_connected(net, 512, bn=True,
        is_training=is_training, scope='rp-fc2', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7,
    #    is_training=is_training, scope='rp-dp2')
    # The first NUM_CENTER_BIN*2*2: CENTER_BIN class scores and bin residuals for (x,z)
    # next 1: center residual for y
    # next NUM_HEADING_BIN*2: heading bin class scores and residuals
    # next NUM_SIZE_CLUSTER*4: size cluster class scores and residuals(l,w,h)
    output = tf_util.fully_connected(net,
        NUM_CENTER_BIN*2*2+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4,
        activation_fn=None, scope='rp-fc3')
    end_points['proposals'] = output
    return output

def get_model(point_cloud, labels_pl, is_training, bn_decay, end_points):
    mask_label = labels_pl['mask_label']
    end_points = get_segmentation_net(point_cloud, is_training, bn_decay, end_points)
    foreground_logits = tf.cond(is_training, lambda: tf.one_hot(mask_label, 2), lambda: end_points['foreground_logits'])
    # fg_point_feats include xyz
    fg_point_feats, end_points = point_cloud_masking(
        end_points['point_feats'], foreground_logits,
        end_points, xyz_only=False) # BxNUM_FG_POINTxD
    proposals = get_region_proposal_net(fg_point_feats, is_training, bn_decay, end_points)
    batch_size = fg_point_feats.get_shape()[0].value
    proposals_reshaped = tf.reshape(proposals, [batch_size, NUM_FG_POINT, -1])
    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(proposals_reshaped, end_points)
    end_points = reduce_proposals(end_points)
    # for iou eval
    end_points['gt_box_of_point'] = tf.gather_nd(labels_pl['gt_box_of_point'], end_points['fg_point_indices'])
    end_points['gt_box_of_point'].set_shape([batch_size, NUM_FG_POINT, 8, 3])
    return end_points

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
