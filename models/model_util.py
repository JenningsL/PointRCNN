import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from parameterize import NUM_CENTER_BIN, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, CENTER_SEARCH_RANGE, CENTER_BIN_SIZE, type_mean_size
from parameterize import class2angle, class2size, class2center
from box_util import get_3d_box
import tf_util
from tensorflow.python.ops import array_ops

# -----------------
# Global Constants
# -----------------
NUM_FG_POINT = 1024
NUM_CHANNEL = 4

# -----------------
# TF Functions Helpers
# -----------------

def tf_gather_object_pc(point_cloud, mask, npoints=512):
    ''' Gather object point clouds according to predicted masks.
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    '''
    def mask_to_indices(mask):
        indices = np.zeros((mask.shape[0], npoints, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i,:]>0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > npoints:
                    choice = np.random.choice(len(pos_indices),
                        npoints, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                        npoints-len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i,:,1] = pos_indices[choice]
            indices[i,:,0] = i
        return indices

    indices = tf.py_func(mask_to_indices, [mask], tf.int32)
    object_pc = tf.gather_nd(point_cloud, indices)
    return object_pc, indices


def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    #print '-----', centers
    N = centers.get_shape()[0].value
    l = tf.slice(sizes, [0,0], [-1,1]) # (N,1)
    w = tf.slice(sizes, [0,1], [-1,1]) # (N,1)
    h = tf.slice(sizes, [0,2], [-1,1]) # (N,1)
    #print l,w,h
    x_corners = tf.concat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], axis=1) # (N,8)
    y_corners = tf.concat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], axis=1) # (N,8)
    z_corners = tf.concat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], axis=1) # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners,1), tf.expand_dims(y_corners,1), tf.expand_dims(z_corners,1)], axis=1) # (N,3,8)
    #print x_corners, y_corners, z_corners
    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c,zeros,s], axis=1) # (N,3)
    row2 = tf.stack([zeros,ones,zeros], axis=1)
    row3 = tf.stack([-s,zeros,c], axis=1)
    R = tf.concat([tf.expand_dims(row1,1), tf.expand_dims(row2,1), tf.expand_dims(row3,1)], axis=1) # (N,3,3)
    #print row1, row2, row3, R, N
    corners_3d = tf.matmul(R, corners) # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers,2), [1,1,8]) # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0,2,1]) # (N,8,3)
    return corners_3d

def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.get_shape()[0].value
    heading_bin_centers = tf.constant(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN), dtype=tf.float32) # (NH,)
    headings = heading_residuals + tf.expand_dims(heading_bin_centers, 0) # (B,NH)

    mean_sizes = tf.expand_dims(tf.constant(type_mean_size, dtype=tf.float32), 0) + size_residuals # (B,NS,1)
    sizes = mean_sizes + size_residuals # (B,NS,3)
    sizes = tf.tile(tf.expand_dims(sizes,1), [1,NUM_HEADING_BIN,1,1]) # (B,NH,NS,3)
    headings = tf.tile(tf.expand_dims(headings,-1), [1,1,NUM_SIZE_CLUSTER]) # (B,NH,NS)
    centers = tf.tile(tf.expand_dims(tf.expand_dims(center,1),1), [1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1]) # (B,NH,NS,3)

    N = batch_size*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N,3]), tf.reshape(headings, [N]), tf.reshape(sizes, [N,3]))

    return tf.reshape(corners_3d, [batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3])

def get_3d_box_from_output(end_points):
    """ Parse tensor output to 3d box
    Inputs:
        end_points['fg_points_xyz'] # (B,N,3)
        end_points['center_x_scores'] # (B,N)
        end_points['center_x_residuals_normalized'] # (B,N,NUM_CENTER_BIN)
        end_points['center_z_scores'] # (B,N)
        end_points['center_z_residuals_normalized'] # (B,N,NUM_CENTER_BIN)
        end_points['center_y_residuals'] # (B,N,1)
        end_points['heading_scores'] # (B,N,NUM_HEADING_BIN)
        end_points['heading_residuals_normalized'] # (B,N,NUM_HEADING_BIN)
        end_points['size_scores'] # (B,N,NUM_SIZE_CLUSTER)
        end_points['size_residuals_normalized'] # (B,N,NUM_SIZE_CLUSTER,3)
    Outputs:
        center: (B,N,3) tensor
        angle: (B,N) tensor
        box_size: (B,N,3) tensor
    """
    batch_size = end_points['fg_points_xyz'].shape[0]
    fg_points_num = end_points['fg_points_xyz'].shape[1]
    # center
    bin_x = tf.argmax(end_points['center_x_scores'], axis=-1) # (B,N)
    bin_x_onehot = tf.one_hot(bin_x, depth=NUM_CENTER_BIN, axis=-1) # (B,N,NUM_CENTER_BIN)
    center_x_residuals_normalized = tf.reduce_sum(
        end_points['center_x_residuals_normalized']*tf.to_float(bin_x_onehot), axis=2) # BxN
    bin_z = tf.argmax(end_points['center_z_scores'], axis=-1) # (B,N)
    bin_z_onehot = tf.one_hot(bin_z, depth=NUM_CENTER_BIN, axis=-1) # (B,N,NUM_CENTER_BIN)
    center_z_residuals_normalized = tf.reduce_sum(
        end_points['center_z_residuals_normalized']*tf.to_float(bin_z_onehot), axis=2) # BxN
    center_y_residuals = end_points['center_y_residuals']
    points_xyz = end_points['fg_points_xyz']
    bin_size = tf.constant(CENTER_BIN_SIZE, dtype=tf.float32)
    search_range = tf.constant(CENTER_SEARCH_RANGE, dtype=tf.float32)

    bin_center = tf.stack([
        tf.to_float(bin_x) * bin_size + bin_size/2 - search_range + tf.gather(points_xyz, 0, axis=-1),
        tf.gather(points_xyz, 1, axis=-1),
        tf.to_float(bin_z) * bin_size + bin_size/2 - search_range + tf.gather(points_xyz, 2, axis=-1)
    ], axis=-1) # (B,N,3)
    center_res = tf.stack([
        center_x_residuals_normalized * bin_size,
        tf.gather(center_y_residuals, 0, axis=-1),
        #center_y_residuals,
        center_z_residuals_normalized * bin_size
    ], axis=-1)
    center = bin_center + center_res # (B,N,3)
    # heading
    heading_cls = tf.argmax(end_points['heading_scores'], axis=-1)
    heading_one_hot = tf.one_hot(heading_cls, depth=NUM_HEADING_BIN, axis=-1)
    heading_res_norm = tf.reduce_sum(end_points['heading_residuals_normalized']*tf.to_float(heading_one_hot), axis=2) # BxN
    angle_per_class = tf.constant(2*np.pi/float(NUM_HEADING_BIN), dtype=tf.float32)
    angle = tf.to_float(heading_cls) * angle_per_class + heading_res_norm * angle_per_class
    # to label format
    angle = tf.where(angle > np.pi, angle - 2*np.pi, angle) # (B,N)
    # size
    size_cls = tf.argmax(end_points['size_scores'], axis=-1)
    size_one_hot = tf.one_hot(size_cls, depth=NUM_SIZE_CLUSTER, axis=-1)
    size_one_hot_tiled = tf.tile(
        tf.expand_dims(tf.to_float(size_one_hot), -1), [1,1,1,3]) # BxNxNUM_SIZE_CLUSTERx3
    size_res_norm = tf.reduce_sum(
        end_points['size_residuals_normalized']*size_one_hot_tiled, axis=2) # BxNx3
    mean_size_arr_expand = tf.expand_dims(tf.expand_dims( \
        tf.constant(type_mean_size, dtype=tf.float32),0), 0) # NUM_SIZE_CLUSTERx3 -> 1x1xNUM_SIZE_CLUSTERx3
    mean_size_arr_expand_tiled = tf.tile(mean_size_arr_expand, [batch_size, fg_points_num, 1, 1])
    mean_size = tf.reduce_sum( \
        size_one_hot_tiled * mean_size_arr_expand_tiled, axis=2) # BxNx3
    size_res = size_res_norm * mean_size
    box_size = mean_size + size_res # (B,N,3)

    return center, angle, box_size

def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)

def parse_output_to_tensors(output, end_points):
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
    box_center, box_angle, box_size = get_3d_box_from_output(end_points)
    box_num = batch_size * npoints
    corners_3d = get_box3d_corners_helper(
        tf.reshape(box_center, [box_num,3]), tf.reshape(box_angle, [box_num]), tf.reshape(box_size, [box_num,3]))
    end_points['proposal_boxes'] = tf.reshape(corners_3d, [batch_size, npoints, 8, 3])
    return end_points

# --------------------------------------
# Shared subgraphs for v1 and v2 models
# --------------------------------------

def placeholder_inputs(batch_size, num_point):
    ''' Get useful placeholder tensors.
    Input:
        batch_size: scalar int
        num_point: scalar int
    Output:
        TF placeholders for inputs and ground truths
    '''
    pointclouds_pl = tf.placeholder(tf.float32,
        shape=(batch_size, num_point, NUM_CHANNEL))
    seg_labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    center_bin_x_labels = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    center_bin_z_labels = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    center_x_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    center_z_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    center_y_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    heading_bin_labels = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    heading_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    size_class_labels = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    size_residuals_labels = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    gt_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 8, 3))
    gt_box_of_point = tf.placeholder(tf.float32, shape=(batch_size, num_point, 8, 3))
    return pointclouds_pl, seg_labels_pl, center_bin_x_labels, center_bin_z_labels,\
        center_x_residuals_labels, center_z_residuals_labels, center_y_residuals_labels, heading_bin_labels,\
        heading_residuals_labels, size_class_labels, size_residuals_labels, gt_boxes, gt_box_of_point


def point_cloud_masking(point_cloud, logits, end_points, xyz_only=True):
    ''' Select point cloud with predicted 3D mask
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        logits: TF tensor in shape (B,N,2)
        end_points: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: TF tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_FG_POINT as a hyper-parameter
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    mask = tf.slice(logits,[0,0,0],[-1,-1,1]) < \
        tf.slice(logits,[0,0,1],[-1,-1,1])
    mask = tf.to_float(mask) # BxNx1
    mask_count = tf.tile(tf.reduce_sum(mask,axis=1,keep_dims=True),
        [1,1,3]) # Bx1x3
    point_cloud_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) # BxNx3
    # mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1,1,3])*point_cloud_xyz,
    #     axis=1, keep_dims=True) # Bx1x3
    mask = tf.squeeze(mask, axis=[2]) # BxN
    end_points['mask'] = mask
    # mask_xyz_mean = mask_xyz_mean/tf.maximum(mask_count,1) # Bx1x3

    # Translate to masked points' centroid
    # point_cloud_xyz_stage1 = point_cloud_xyz - \
    #     tf.tile(mask_xyz_mean, [1,num_point,1])
    point_cloud_xyz_stage1 = point_cloud_xyz
    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])
        point_cloud_stage1 = tf.concat(\
            [point_cloud_xyz_stage1, point_cloud_features], axis=-1)
    num_channels = point_cloud_stage1.get_shape()[2].value

    object_point_cloud, indices = tf_gather_object_pc(point_cloud_stage1,
        mask, NUM_FG_POINT)
    object_point_cloud.set_shape([batch_size, NUM_FG_POINT, num_channels])
    end_points['fg_point_indices'] = indices
    end_points['fg_points'] = object_point_cloud
    end_points['fg_points_xyz'] = tf.slice(object_point_cloud, [0,0,0], [-1,-1,3])

    return object_point_cloud, end_points


def get_center_regression_net(object_point_cloud, one_hot_vec,
                              is_training, bn_decay, end_points):
    ''' Regression network for center delta. a.k.a. T-Net.
    Input:
        object_point_cloud: TF tensor in shape (B,M,C)
            point clouds in 3D mask coordinate
        one_hot_vec: TF tensor in shape (B,3)
            length-3 vectors indicating predicted object type
    Output:
        predicted_center: TF tensor in shape (B,3)
    '''
    num_point = object_point_cloud.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2-stage1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3-stage1', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
        padding='VALID', scope='maxpool-stage1')
    net = tf.squeeze(net, axis=[1,2])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 256, scope='fc1-stage1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, scope='fc2-stage1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    predicted_center = tf_util.fully_connected(net, 3, activation_fn=None,
        scope='fc3-stage1')
    return predicted_center, end_points


def get_loss(labels, end_points):
    batch_size = end_points['foreground_logits'].get_shape()[0].value
    #npoints = end_points['foreground_logits'].get_shape()[1].value
    # 3D Segmentation loss
    #mask_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
    #   logits=end_points['foreground_logits'], labels=labels['mask_label']))
    mask_loss = focal_loss(end_points['foreground_logits'], tf.one_hot(labels['mask_label'], 2, axis=-1))
    tf.summary.scalar('mask loss', mask_loss)
    props = end_points['proposals']
    # TODO: mask foreground points
    labels_fg = {}
    for k in labels.keys():
        if k in ['mask_label']:
            continue
        labels_fg[k] = tf.gather_nd(labels[k], end_points['fg_point_indices'])
        if k == 'size_residuals':
            labels_fg[k].set_shape([batch_size, NUM_FG_POINT, 3])
        else:
            labels_fg[k].set_shape([batch_size, NUM_FG_POINT])
    # Center loss
    #objectness_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
    #   logits=end_points['objectness'], labels=labels_fg['objectness']))
    #tf.summary.scalar('objectness loss', objectness_loss)
    center_x_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
       logits=end_points['center_x_scores'], labels=labels_fg['center_bin_x']))
    center_z_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\
       logits=end_points['center_z_scores'], labels=labels_fg['center_bin_z']))
    bin_x_onehot = tf.one_hot(labels_fg['center_bin_x'],
        depth=NUM_CENTER_BIN,
        on_value=1, off_value=0, axis=-1) # BxNxNUM_CENTER_BIN
    # NOTICE: labels['center_x_residuals'] is already normalized
    center_x_residuals_normalized = tf.reduce_sum(end_points['center_x_residuals_normalized']*tf.to_float(bin_x_onehot), axis=2) # BxN
    center_x_residuals_dist = tf.norm(labels_fg['center_x_residuals'] - center_x_residuals_normalized, axis=-1)
    center_x_res_loss = huber_loss(center_x_residuals_dist, delta=1.0)
    bin_z_onehot = tf.one_hot(labels_fg['center_bin_z'],
        depth=NUM_CENTER_BIN,
        on_value=1, off_value=0, axis=-1) # BxNxNUM_CENTER_BIN
    center_z_residuals_normalized = tf.reduce_sum(end_points['center_z_residuals_normalized']*tf.to_float(bin_z_onehot), axis=2) # BxN
    center_z_residuals_dist = tf.norm(labels_fg['center_z_residuals'] - center_z_residuals_normalized, axis=-1)
    center_z_res_loss = huber_loss(center_z_residuals_dist, delta=1.0)
    # y is directly regressed
    center_y_residuals_dist = tf.norm(labels_fg['center_y_residuals'] - tf.squeeze(end_points['center_y_residuals']), axis=-1)
    center_y_res_loss = huber_loss(center_y_residuals_dist, delta=1.0)
    tf.summary.scalar('center_x  class loss', center_x_cls_loss)
    tf.summary.scalar('center_z  class loss', center_z_cls_loss)
    tf.summary.scalar('center_x residual loss', center_x_res_loss)
    tf.summary.scalar('center_y residual loss', center_y_res_loss)
    tf.summary.scalar('center_z residual loss', center_z_res_loss)
    # Heading loss
    heading_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['heading_scores'], labels=labels_fg['heading_bin']))
    hcls_onehot = tf.one_hot(labels_fg['heading_bin'],
        depth=NUM_HEADING_BIN,
        on_value=1, off_value=0, axis=-1) # BxNxNUM_HEADING_BIN
    heading_residual_normalized_label = \
        labels_fg['heading_residuals'] / (2*np.pi/float(NUM_HEADING_BIN))
    heading_res_loss = huber_loss(tf.reduce_sum( \
        end_points['heading_residuals_normalized']*tf.to_float(hcls_onehot), axis=2) - \
        heading_residual_normalized_label, delta=1.0)
    tf.summary.scalar('heading class loss', heading_class_loss)
    tf.summary.scalar('heading residual loss', heading_res_loss)
    # Size loss
    size_class_loss = tf.reduce_mean( \
        tf.nn.sparse_softmax_cross_entropy_with_logits( \
        logits=end_points['size_scores'], labels=labels_fg['size_class']))

    scls_onehot = tf.one_hot(labels_fg['size_class'],
        depth=NUM_SIZE_CLUSTER,
        on_value=1, off_value=0, axis=-1) # BxNxNUM_SIZE_CLUSTER
    scls_onehot_tiled = tf.tile(tf.expand_dims( \
        tf.to_float(scls_onehot), -1), [1,1,1,3]) # BxNxNUM_SIZE_CLUSTERx3
    predicted_size_residual_normalized = tf.reduce_sum( \
        end_points['size_residuals_normalized']*scls_onehot_tiled, axis=2) # BxNx3

    mean_size_arr_expand = tf.expand_dims(tf.expand_dims( \
        tf.constant(type_mean_size, dtype=tf.float32),0), 0) # NUM_SIZE_CLUSTERx3 -> 1x1xNUM_SIZE_CLUSTERx3
    mean_size_arr_expand_tiled = tf.tile(mean_size_arr_expand, [batch_size, NUM_FG_POINT, 1, 1])
    mean_size_label = tf.reduce_sum( \
        scls_onehot_tiled * mean_size_arr_expand_tiled, axis=2) # BxNx3
    size_residual_label_normalized = labels_fg['size_residuals'] / mean_size_label # BxNx3

    size_dist = tf.norm(size_residual_label_normalized - predicted_size_residual_normalized, axis=-1)
    size_res_loss = huber_loss(size_dist, delta=1.0)
    tf.summary.scalar('size class loss', size_class_loss)
    tf.summary.scalar('size residual loss', size_res_loss)

    seg_weight = 0.01
    cls_weight = 1
    res_weight = 1
    total_loss = seg_weight * mask_loss + \
        cls_weight * (center_x_cls_loss + center_z_cls_loss + heading_class_loss + 10*size_class_loss) + \
        res_weight * (center_x_res_loss + center_z_res_loss + center_y_res_loss + 100*heading_res_loss + 100*size_res_loss)
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

if __name__ == '__main__':
    placeholder_inputs(32, 1024)
    sys.exit()
    import math
    from parameterize import obj_to_proposal_vec, CENTER_BIN_SIZE
    class Box(object):
        pass
    obj = Box()
    obj.t = np.array([-28.3,1.2,5.2])
    obj.ry = math.pi * 0.3
    obj.l = 1
    obj.w = 0.6
    obj.h = 0.7
    point = np.array([-27.1,1.3,4.3])
    center_cls, center_res, angle_cls,angle_res, size_cls, size_res = obj_to_proposal_vec(obj, point)
    angle_res /= (2*np.pi/float(NUM_HEADING_BIN))
    size_res /= type_mean_size[size_cls]
    # print(center_cls, center_res, angle_cls,angle_res, size_cls, size_res)
    # print(class2center(center_cls, center_res, point))
    # print(class2size(size_cls, size_res))
    # print(class2angle(angle_cls,angle_res, NUM_HEADING_BIN))

    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(np.array(targets).shape)+[nb_classes])[0]

    center_x_scores = get_one_hot(center_cls[0:1],NUM_CENTER_BIN)
    center_x_residuals_normalized = np.tile(center_res[:1], NUM_CENTER_BIN)
    center_z_scores = get_one_hot(center_cls[1:],NUM_CENTER_BIN)
    center_z_residuals_normalized = np.tile(center_res[2:], NUM_CENTER_BIN)
    heading_scores = get_one_hot([angle_cls], NUM_HEADING_BIN)
    heading_residuals_normalized = np.tile([angle_res], NUM_HEADING_BIN)
    size_scores = get_one_hot([size_cls], NUM_SIZE_CLUSTER)
    size_residuals_normalized = np.tile([size_res], (NUM_SIZE_CLUSTER,1))
    centers, angles, sizes = get_3d_box_from_output({
        'fg_points_xyz': tf.constant(np.array([[point]]), dtype=tf.float32),
        'center_x_scores': tf.constant(np.array([[center_x_scores]]), dtype=tf.float32),
        'center_x_residuals_normalized': tf.constant(np.array([[center_x_residuals_normalized]]), dtype=tf.float32),
        'center_z_scores': tf.constant(np.array([[center_z_scores]]), dtype=tf.float32),
        'center_z_residuals_normalized': tf.constant(np.array([[center_z_residuals_normalized]]), dtype=tf.float32),
        'center_y_residuals': tf.constant(np.array([[center_res[1:2]]]), dtype=tf.float32),
        'heading_scores': tf.constant(np.array([[heading_scores]]), dtype=tf.float32),
        'heading_residuals_normalized': tf.constant(np.array([[heading_residuals_normalized]]), dtype=tf.float32),
        'size_scores': tf.constant(np.array([[size_scores]]), dtype=tf.float32),
        'size_residuals_normalized': tf.constant(np.array([[size_residuals_normalized]]), dtype=tf.float32)
    })
    N = 1 * 1 # batch * num_point
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N,3]), tf.reshape(angles, [N]), tf.reshape(sizes, [N,3]))
    with tf.Session() as sess:
        c, a, s = sess.run([centers, angles, sizes])
        print(c[0])
        print(a[0])
        print(s[0])
        corners_list = sess.run(corners_3d)
        print(corners_list)
