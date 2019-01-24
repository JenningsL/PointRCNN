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
from parameterize import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_CENTER_BIN, NUM_OBJ_CLASSES, type_mean_size
from model_util import huber_loss

class RCNN(object):
    def __init__(self, batch_size, num_point, num_channel=133, bn_decay=None):
        self.batch_size = batch_size
        self.num_point = num_point
        self.num_channel = num_channel
        self.bn_decay = bn_decay
        self.end_points = {}
        self.placeholders = self.get_placeholders()
        self.build()

    def get_placeholders(self):
        batch_size = self.batch_size
        num_point = self.num_point
        num_channel = self.num_channel
        pointclouds_pl = tf.placeholder(tf.float32,
            shape=(batch_size, num_point, num_channel))
        proposal_centers_pl = tf.placeholder(tf.float32, shape=(batch_size, 3))
        class_labels_pl = tf.placeholder(tf.int32, shape=(batch_size,))
        center_bin_x_labels = tf.placeholder(tf.int32, shape=(batch_size,))
        center_bin_z_labels = tf.placeholder(tf.int32, shape=(batch_size,))
        center_x_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size,))
        center_z_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size,))
        center_y_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size,))
        heading_bin_labels = tf.placeholder(tf.int32, shape=(batch_size,))
        heading_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size,))
        size_class_labels = tf.placeholder(tf.int32, shape=(batch_size,))
        size_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size, 3))
        gt_box_of_prop = tf.placeholder(tf.float32, shape=(batch_size, 8, 3))
        is_training_pl = tf.placeholder(tf.bool, shape=())
        return {
            'pointclouds': pointclouds_pl,
            'proposal_centers': proposal_centers_pl,
            'class_labels': class_labels_pl,
            'center_bin_x_labels': center_bin_x_labels,
            'center_bin_z_labels': center_bin_z_labels,
            'center_x_res_labels': center_x_residuals_labels,
            'center_z_res_labels': center_z_residuals_labels,
            'center_y_res_labels': center_y_residuals_labels,
            'heading_bin_labels': heading_bin_labels,
            'heading_res_labels': heading_residuals_labels,
            'size_class_labels': size_class_labels,
            'size_res_labels': size_residuals_labels,
            'gt_box_of_prop': gt_box_of_prop,
            'is_training_pl': is_training_pl
        }

    def build(self):
        point_cloud = self.placeholders['pointclouds']
        is_training = self.placeholders['is_training_pl']
        batch_size = self.batch_size
        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,self.num_channel-3])
        # Set abstraction layers
        l1_xyz, l1_points, _ = pointnet_sa_module(l0_xyz, l0_points,
            npoint=128, radius=0.2, nsample=64, mlp=[128,128,128],
            mlp2=None, group_all=False, is_training=is_training, bn_decay=self.bn_decay,
            scope='rcnn-sa1', bn=True)
        l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points,
            npoint=64, radius=0.4, nsample=64, mlp=[128,128,256],
            mlp2=None, group_all=False, is_training=is_training, bn_decay=self.bn_decay,
            scope='rcnn-sa2', bn=True)
        l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
            npoint=64, radius=0.4, nsample=64, mlp=[256,256,512],
            mlp2=None, group_all=True, is_training=is_training, bn_decay=self.bn_decay,
            scope='rcnn-sa3', bn=True)

        feats = tf.reshape(l3_points, [batch_size, -1])

        # Classification
        cls_net = tf_util.fully_connected(feats, 512, bn=True, is_training=is_training, scope='rcnn-cls-fc1', bn_decay=self.bn_decay)
        #cls_net = tf_util.dropout(cls_net, keep_prob=0.4, is_training=is_training, scope='cls_dp1')
        cls_net = tf_util.fully_connected(cls_net, 256, bn=True, is_training=is_training, scope='rcnn-cls-fc2', bn_decay=self.bn_decay)
        #cls_net = tf_util.dropout(cls_net, keep_prob=0.4, is_training=is_training, scope='cls_dp2')
        cls_net = tf_util.fully_connected(cls_net, NUM_OBJ_CLASSES, activation_fn=None, scope='rcnn-cls-fc3')
        self.end_points['cls_logits'] = cls_net

        # Box estimation
        net = tf_util.fully_connected(feats, 512, bn=True,
            is_training=is_training, scope='rcnn-fc1', bn_decay=self.bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True,
            is_training=is_training, scope='rcnn-fc2', bn_decay=self.bn_decay)
        # The first NUM_CENTER_BIN*2*2: CENTER_BIN class scores and bin residuals for (x,z)
        # next 1: center residual for y
        # next NUM_HEADING_BIN*2: heading bin class scores and residuals
        # next NUM_SIZE_CLUSTER*4: size cluster class scores and residuals(l,w,h)
        output = tf_util.fully_connected(net,
            NUM_CENTER_BIN*2*2+1+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4,
            activation_fn=None, scope='rcnn-fc3')
        self.parse_output_to_tensors(output)
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
        box_center, box_angle, box_size = get_3d_box_from_output(end_points)
        box_center = tf.squeeze(box_center, axis=1)
        box_center = box_center + self.placeholders['proposal_centers']
        box_angle = tf.squeeze(box_angle, axis=1)
        box_size = tf.squeeze(box_size, axis=1)
        corners_3d = get_box3d_corners_helper(box_center, box_angle, box_size)
        self.end_points['output_boxes'] = corners_3d
        return corners_3d

    def get_loss(self):
        end_points = self.end_points
        cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=end_points['cls_logits'], labels=self.placeholders['class_labels']))
        tf.summary.scalar('classification loss', cls_loss)
        is_obj_mask = tf.to_float(tf.not_equal(self.placeholders['class_labels'], 0))
        center_x_cls_loss = tf.reduce_mean(is_obj_mask*tf.nn.sparse_softmax_cross_entropy_with_logits(\
           logits=end_points['center_x_scores'], labels=self.placeholders['center_bin_x_labels']))
        center_z_cls_loss = tf.reduce_mean(is_obj_mask*tf.nn.sparse_softmax_cross_entropy_with_logits(\
           logits=end_points['center_z_scores'], labels=self.placeholders['center_bin_z_labels']))
        bin_x_onehot = tf.one_hot(self.placeholders['center_bin_x_labels'],
            depth=NUM_CENTER_BIN,
            on_value=1, off_value=0, axis=-1) # BxNUM_CENTER_BIN
        # NOTICE: labels['center_x_residuals'] is already normalized
        center_x_residuals_normalized = tf.reduce_sum(end_points['center_x_residuals_normalized']*tf.to_float(bin_x_onehot), axis=-1) # B
        center_x_residuals_dist = tf.norm(self.placeholders['center_x_res_labels'] - center_x_residuals_normalized, axis=-1)
        center_x_res_loss = huber_loss(is_obj_mask*center_x_residuals_dist, delta=1.0)
        bin_z_onehot = tf.one_hot(self.placeholders['center_bin_z_labels'],
            depth=NUM_CENTER_BIN,
            on_value=1, off_value=0, axis=-1) # BxNUM_CENTER_BIN
        center_z_residuals_normalized = tf.reduce_sum(end_points['center_z_residuals_normalized']*tf.to_float(bin_z_onehot), axis=-1) # B
        center_z_residuals_dist = tf.norm(self.placeholders['center_z_res_labels'] - center_z_residuals_normalized, axis=-1)
        center_z_res_loss = huber_loss(is_obj_mask*center_z_residuals_dist, delta=1.0)
        # y is directly regressed
        center_y_residuals_dist = tf.norm(self.placeholders['center_y_res_labels'] - tf.gather(end_points['center_y_residuals'], 0, axis=-1), axis=-1)
        center_y_res_loss = huber_loss(is_obj_mask*center_y_residuals_dist, delta=1.0)
        tf.summary.scalar('center_x  class loss', center_x_cls_loss)
        tf.summary.scalar('center_z  class loss', center_z_cls_loss)
        tf.summary.scalar('center_x residual loss', center_x_res_loss)
        tf.summary.scalar('center_y residual loss', center_y_res_loss)
        tf.summary.scalar('center_z residual loss', center_z_res_loss)
        # Heading loss
        heading_class_loss = tf.reduce_mean( \
            is_obj_mask*tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['heading_scores'], labels=self.placeholders['heading_bin_labels']))
        hcls_onehot = tf.one_hot(self.placeholders['heading_bin_labels'],
            depth=NUM_HEADING_BIN,
            on_value=1, off_value=0, axis=-1) # BxNxNUM_HEADING_BIN
        heading_residual_normalized_label = \
            self.placeholders['heading_res_labels'] / (2*np.pi/float(NUM_HEADING_BIN))
        heading_res_dist = tf.norm(tf.reduce_sum( \
            end_points['heading_residuals_normalized']*tf.to_float(hcls_onehot), axis=-1) - \
            heading_residual_normalized_label)
        heading_res_loss = huber_loss(is_obj_mask*heading_res_dist, delta=1.0)
        tf.summary.scalar('heading class loss', heading_class_loss)
        tf.summary.scalar('heading residual loss', heading_res_loss)
        # Size loss
        size_class_loss = tf.reduce_mean( \
            is_obj_mask*tf.nn.sparse_softmax_cross_entropy_with_logits( \
            logits=end_points['size_scores'], labels=self.placeholders['size_class_labels']))

        scls_onehot = tf.one_hot(self.placeholders['size_class_labels'],
            depth=NUM_SIZE_CLUSTER,
            on_value=1, off_value=0, axis=-1) # BxNUM_SIZE_CLUSTER
        scls_onehot_tiled = tf.tile(tf.expand_dims( \
            tf.to_float(scls_onehot), -1), [1,1,3]) # BxNUM_SIZE_CLUSTERx3
        predicted_size_residual_normalized = tf.reduce_sum( \
            end_points['size_residuals_normalized']*scls_onehot_tiled, axis=-1) # Bx3

        mean_size_arr_expand = tf.expand_dims( \
            tf.constant(type_mean_size, dtype=tf.float32),0) # NUM_SIZE_CLUSTERx3 -> 1xNUM_SIZE_CLUSTERx3
        mean_size_arr_expand_tiled = tf.tile(mean_size_arr_expand, [self.batch_size, 1, 1])
        mean_size_label = tf.reduce_sum( \
            scls_onehot_tiled * mean_size_arr_expand_tiled, axis=2) # Bx3
        size_residual_label_normalized = self.placeholders['size_res_labels'] / mean_size_label # Bx3

        size_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
        size_res_loss = huber_loss(is_obj_mask*size_dist, delta=1.0)
        tf.summary.scalar('size class loss', size_class_loss)
        tf.summary.scalar('size residual loss', size_res_loss)

        obj_cls_weight = 1
        cls_weight = 1
        res_weight = 1
        total_loss = obj_cls_weight * cls_loss + \
            cls_weight * (center_x_cls_loss + center_z_cls_loss + heading_class_loss + 10*size_class_loss) + \
            res_weight * (center_x_res_loss + center_z_res_loss + center_y_res_loss + 100*heading_res_loss + 100*size_res_loss)
        return total_loss

if __name__ == '__main__':
    with tf.Graph().as_default():
        model = RCNN(32, 512)
        for key in model.end_points:
            print((key, model.end_points[key]))
