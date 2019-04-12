''' Frustum PointNets v2 Model.
'''
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from frustum_model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from frustum_model_util import NUM_SEG_CLASSES, NUM_OBJ_CLASSES, NUM_CHANNEL
from frustum_model_util import point_cloud_masking, get_center_regression_net
from frustum_model_util import placeholder_inputs, parse_output_to_tensors, get_loss
import projection
# https://github.com/deepsense-ai/roi-pooling
# from roi_pooling.roi_pooling_ops import roi_pooling

class FrustumPointNet(object):
    """docstring for FrustumPointNet."""

    def __init__(self, batch_size, num_point, bn_decay=None):
        super(FrustumPointNet, self).__init__()
        self.batch_size = batch_size
        self.num_point = num_point
        self.bn_decay = bn_decay
        self.placeholders = self.get_placeholders()
        self.end_points = {}
        self.build()

    def get_placeholders(self):
        batch_size = self.batch_size
        num_point = self.num_point
        return {
            'pointclouds': tf.placeholder(tf.float32, shape=(batch_size, num_point, NUM_CHANNEL)),
            'img_seg_map': tf.placeholder(tf.float32, shape=(batch_size, 360, 1200, 4)), # TODO: don't hardcode feature_vector size
            'prop_box': tf.placeholder(tf.float32, shape=(batch_size, 7)),
            'calib': tf.placeholder(tf.float32, shape=(batch_size, 3, 4)),
            'cls_label': tf.placeholder(tf.int32, shape=(batch_size,)),
            'ious': tf.placeholder(tf.float32, shape=(batch_size,)),
            'seg_labels': tf.placeholder(tf.int32, shape=(batch_size, num_point)),
            'centers': tf.placeholder(tf.float32, shape=(batch_size, 3)),
            'heading_class_label': tf.placeholder(tf.int32, shape=(batch_size,)),
            'heading_residual_label': tf.placeholder(tf.float32, shape=(batch_size,)),
            'size_class_label': tf.placeholder(tf.int32, shape=(batch_size,)),
            'size_residual_label': tf.placeholder(tf.float32, shape=(batch_size,3)),
            'is_training': tf.placeholder(tf.bool, shape=())
        }

    def build(self):
        ''' Frustum PointNets model. The model predict 3D object masks and
        amodel bounding boxes for objects in frustum point clouds.

        Input:
            point_cloud: TF tensor in shape (B,N,4)
                frustum point clouds with XYZ and intensity in point channels
                XYZs are in frustum coordinate
            feature_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
            is_training: TF boolean scalar
            bn_decay: TF float scalar
        Output:
            end_points: dict (map from name strings to TF tensors)
        '''
        end_points = self.end_points
        batch_size = self.batch_size
        point_cloud = self.placeholders['pointclouds']
        cls_label = self.placeholders['cls_label']
        img_seg_map = self.placeholders['img_seg_map']
        proposal_boxes = self.placeholders['prop_box']
        calib = self.placeholders['calib']
        is_training = self.placeholders['is_training']
        bn_decay = self.bn_decay

        with tf.variable_scope('proposal_classification'):
            end_points = self.get_proposal_cls_net(point_cloud, img_seg_map, proposal_boxes, calib, is_training)

        cls_label_pred = tf.argmax(tf.nn.softmax(end_points['cls_logits']), axis=1)
        end_points['one_hot_vec'] = tf.one_hot(cls_label_pred, NUM_OBJ_CLASSES)
        end_points['one_hot_gt'] = tf.one_hot(cls_label, NUM_OBJ_CLASSES)
        one_hot_vec = tf.cond(is_training, lambda: end_points['one_hot_gt'], lambda: end_points['one_hot_vec'])

        with tf.variable_scope('box_regression'):
            # 3D Instance Segmentation PointNet
            logits, end_points = self.get_instance_seg_v2_net(\
                point_cloud, one_hot_vec, is_training)
            end_points['mask_logits'] = logits

            # Masking
            # select masked points and translate to masked points' centroid
            object_point_cloud, mask_xyz_mean, end_points = \
                point_cloud_masking(point_cloud, logits, end_points, xyz_only=False)
            object_point_cloud_xyz = tf.slice(object_point_cloud, [0,0,0], [-1,-1,3])

            # T-Net and coordinate translation
            center_delta, end_points = get_center_regression_net(\
                object_point_cloud, one_hot_vec,
                is_training, bn_decay, end_points)
            stage1_center = center_delta + mask_xyz_mean # Bx3
            end_points['stage1_center'] = stage1_center
            # Get object point cloud in object coordinate
            object_point_cloud_xyz_new = \
                object_point_cloud_xyz - tf.expand_dims(center_delta, 1)
            object_point_cloud_features = tf.slice(object_point_cloud, [0,0,3], [-1,-1,-1])
            object_point_cloud_new = tf.concat([object_point_cloud_xyz_new, object_point_cloud_features], axis=-1)
            # Amodel Box Estimation PointNet
            output, end_points = self.get_3d_box_estimation_v2_net(\
                object_point_cloud_new, one_hot_vec, end_points['img_rois'],
                is_training)

            # Parse output to 3D box parameters
            end_points = parse_output_to_tensors(output, end_points)
            end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

        return end_points


    def get_proposal_cls_net(self, point_cloud, img_seg_map, proposal_boxes, calib, is_training):
        batch_size = self.batch_size
        end_points = self.end_points
        bn_decay = self.bn_decay
        end_points = self.end_points

        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,NUM_CHANNEL-3])

        # Set abstraction layers
        l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points,
            128, [0.2,0.4,0.8], [32,64,128],
            [[32,32,64], [64,64,128], [64,96,128]],
            is_training, bn_decay, scope='cls_layer1')
        l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points,
            32, [0.4,0.8,1.6], [64,64,128],
            [[64,64,128], [128,128,256], [128,128,256]],
            is_training, bn_decay, scope='cls_layer2')
        l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
            npoint=None, radius=None, nsample=None, mlp=[128,256,512],
            mlp2=None, group_all=True, is_training=is_training,
            bn_decay=bn_decay, scope='cls_layer3')

        # image feature pooling
        _img_pixel_size = np.asarray([360, 1200])
        box2d_corners, box2d_corners_norm = projection.tf_project_to_image_space(
            proposal_boxes,
            calib, _img_pixel_size)
        # crop and resize
        # y1, x1, y2, x2
        box2d_corners_norm_reorder = tf.stack([
            tf.gather(box2d_corners_norm, 1, axis=-1),
            tf.gather(box2d_corners_norm, 0, axis=-1),
            tf.gather(box2d_corners_norm, 3, axis=-1),
            tf.gather(box2d_corners_norm, 2, axis=-1),
        ], axis=-1)
        img_rois = tf.image.crop_and_resize(
            img_seg_map,
            box2d_corners_norm_reorder,
            tf.range(0, batch_size),
            [16,16])
        '''
        # roi pooling in faster-rcnn
        img_seg_map = tf_util.conv2d(img_seg_map, 1, [1,1], padding='VALID', bn=True,
            is_training=is_training, scope='cls_feature_bottleneck', bn_decay=bn_decay)
        # [feature map index, upper left, bottom right coordinates]
        roi_crops = tf.concat([tf.expand_dims(tf.range(0, batch_size), axis=-1), tf.cast(box2d_corners, tf.int32)], axis=-1)
        #print(roi_crops.shape)
        img_rois = roi_pooling(img_seg_map, roi_crops, pool_height=16, pool_width=16)
        img_rois.set_shape([batch_size, 16, 16])
        #print(img_rois.shape)
        '''

        img_feats = tf.reshape(img_rois, [batch_size, -1])
        end_points['img_rois'] = img_feats

        # classification
        point_feats = tf.reshape(l3_points, [batch_size, -1])
        # use image only
        #cls_net = img_feats
        # use point and image feature
        cls_net = tf.concat([point_feats, img_feats], axis=1)
        # use point only
        #cls_net = point_feats
        cls_net = tf_util.fully_connected(cls_net, 512, bn=True, is_training=is_training, scope='cls_fc1', bn_decay=bn_decay)
        cls_net = tf_util.dropout(cls_net, keep_prob=0.5, is_training=is_training, scope='cls_dp1')
        cls_net = tf_util.fully_connected(cls_net, 256, bn=True, is_training=is_training, scope='cls_fc2', bn_decay=bn_decay)
        cls_net = tf_util.dropout(cls_net, keep_prob=0.5, is_training=is_training, scope='cls_dp2')
        cls_net = tf_util.fully_connected(cls_net, NUM_OBJ_CLASSES, activation_fn=None, scope='cls_logits')
        end_points['cls_logits'] = cls_net
        return end_points

    def get_instance_seg_v2_net(self, point_cloud, one_hot_vec, is_training):
        ''' 3D instance segmentation PointNet v2 network.
        Input:
            point_cloud: TF tensor in shape (B,N,4)
                frustum point clouds with XYZ and intensity in point channels
                XYZs are in frustum coordinate
            one_hot_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
            is_training: TF boolean scalar
        Output:
            logits: TF tensor in shape (B,N,2), scores for bkg/clutter and object
            end_points: dict
        '''
        bn_decay = self.bn_decay
        end_points = self.end_points
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
        l3_points = tf.concat([l3_points, tf.expand_dims(one_hot_vec, 1)], axis=2)

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
        end_points['feats'] = net
        net = tf_util.dropout(net, keep_prob=0.7,
            is_training=is_training, scope='dp1')
        logits = tf_util.conv1d(net, 2, 1,
            padding='VALID', activation_fn=None, scope='conv1d-fc2')

        return logits, end_points

    def get_3d_box_estimation_v2_net(self, object_point_cloud, one_hot_vec, feature_vec,
                                     is_training):
        ''' 3D Box Estimation PointNet v2 network.
        Input:
            object_point_cloud: TF tensor in shape (B,M,C)
                masked point clouds in object coordinate
            one_hot_vec: TF tensor in shape (B,3)
                length-3 vectors indicating predicted object type
            feature_vec: ROI feature crop
        Output:
            output: TF tensor in shape (B,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
                including box centers, heading bin class scores and residuals,
                and size cluster scores and residuals
        '''
        # Gather object points
        batch_size = self.batch_size
        bn_decay = self.bn_decay
        end_points = self.end_points

        #l0_xyz = object_point_cloud
        #l0_points = None
        l0_xyz = tf.slice(object_point_cloud, [0,0,0], [-1,-1,3])
        #l0_points = tf.slice(object_point_cloud, [0,0,3], [-1,-1,-1])
        l0_points = None
        # Set abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
            npoint=128, radius=0.2, nsample=64, mlp=[64,64,128],
            mlp2=None, group_all=False,
            is_training=is_training, bn_decay=bn_decay, scope='ssg-layer1')
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
            npoint=32, radius=0.4, nsample=64, mlp=[128,128,256],
            mlp2=None, group_all=False,
            is_training=is_training, bn_decay=bn_decay, scope='ssg-layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
            npoint=None, radius=None, nsample=None, mlp=[256,256,512],
            mlp2=None, group_all=True,
            is_training=is_training, bn_decay=bn_decay, scope='ssg-layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])
        #net = tf.concat([net, one_hot_vec, feature_vec], axis=1)
        net = tf.concat([net, one_hot_vec], axis=1)
        net = tf_util.fully_connected(net, 512, bn=True,
            is_training=is_training, scope='fc1', bn_decay=bn_decay)
        #net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='est_dp1')
        net = tf_util.fully_connected(net, 256, bn=True,
            is_training=is_training, scope='fc2', bn_decay=bn_decay)
        #net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='est_dp2')

        # The first 3 numbers: box center coordinates (cx,cy,cz),
        # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
        # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
        output = tf_util.fully_connected(net,
            3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
        return output, end_points

    def get_loss(self):
        pls = self.placeholders
        return get_loss(pls['cls_label'], pls['ious'], pls['seg_labels'],
            pls['centers'], pls['heading_class_label'], pls['heading_residual_label'],
            pls['size_class_label'], pls['size_residual_label'], self.end_points)


if __name__=='__main__':
    with tf.Graph().as_default():
        net = FrustumPointNet(32, 512)
        net.get_placeholders()
        net.get_loss()
