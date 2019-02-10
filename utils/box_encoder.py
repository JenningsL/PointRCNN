import tensorflow as tf
import numpy as np
import math

# Car, Pedestrian, Cyclist
type_mean_size = np.array([[3.88311640418,1.62856739989,1.52563191462],
    [0.84422524,0.66068622,1.76255119],
    [1.76282397,0.59706367,1.73698127]])
NUM_SIZE_CLUSTER = type_mean_size.shape[0] # one cluster for each type
NUM_OBJ_CLASSES = 4

class BoxEncoder(object):
    """Encode and decode 3d box"""
    def __init__(self, center_range=3.0, num_center_bin=12, heading_range=np.pi, num_heading_bin=12):
        self.CENTER_SEARCH_RANGE = center_range
        self.HEADING_SEARCH_RANGE = heading_range
        self.NUM_CENTER_BIN = num_center_bin
        self.NUM_HEADING_BIN = num_heading_bin
        self.CENTER_BIN_SIZE = center_range * 2 / num_center_bin # in one direction

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual.
        Input:
            angle: rad scalar, from 0-2pi (or -pi~pi), class center at
                0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        Output:
            class_id, int, among 0,1,...,N-1
            residual_angle_norm: float, a number such that
                class*(2pi/N) + residual_angle*angle_per_class = angle
        '''
        assert(angle>=-self.HEADING_SEARCH_RANGE and angle<=self.HEADING_SEARCH_RANGE)
        angle += self.HEADING_SEARCH_RANGE # [-HEADING_SEARCH_RANGE, HEADING_SEARCH_RANGE]->[0, 2*HEADING_SEARCH_RANGE]

        angle_per_class = 2*self.HEADING_SEARCH_RANGE/float(self.NUM_HEADING_BIN)
        shifted_angle = (angle+angle_per_class/2)%(2*self.HEADING_SEARCH_RANGE)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - \
            (class_id * angle_per_class + angle_per_class/2)
        return class_id, residual_angle/angle_per_class

    def class2angle(self, pred_cls, residual):
        ''' Inverse function to angle2class.'''
        angle_per_class = 2*self.HEADING_SEARCH_RANGE/float(self.NUM_HEADING_BIN)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual * angle_per_class
        angle -= self.HEADING_SEARCH_RANGE
        return angle

    def size2class(self, size):
        ''' Convert 3D bounding box size to template class and residuals.
        todo (rqi): support multiple size clusters per type.

        Input:
            size: numpy array of shape (3,) for (l,w,h)
            type_name: string
        Output:
            size_class: int scalar
            size_residual_norm: numpy array of shape (3,)
        '''
        dist = np.linalg.norm(type_mean_size - size, axis=1)
        size_class = np.argmin(dist)
        size_residual = size - type_mean_size[size_class]
        return size_class, size_residual/type_mean_size[size_class]

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class. '''
        mean_size = type_mean_size[pred_cls]
        return mean_size + residual*type_mean_size[pred_cls]

    def center2class(self, obj_center, point):
        center = obj_center - point
        bin_x = int((obj_center[0] - point[0] + self.CENTER_SEARCH_RANGE) / self.CENTER_BIN_SIZE)
        bin_z = int((obj_center[2] - point[2] + self.CENTER_SEARCH_RANGE) / self.CENTER_BIN_SIZE)
        # limit the bin class label range
        bin_x = np.clip(bin_x, 0, self.NUM_CENTER_BIN-1)
        bin_z = np.clip(bin_z, 0, self.NUM_CENTER_BIN-1)
        center_cls = np.array([bin_x, bin_z])
        center_res = np.array([
            1/self.CENTER_BIN_SIZE * (obj_center[0] - point[0] + self.CENTER_SEARCH_RANGE - (bin_x+0.5)*self.CENTER_BIN_SIZE),
            obj_center[1] - point[1],
            1/self.CENTER_BIN_SIZE * (obj_center[2] - point[2] + self.CENTER_SEARCH_RANGE - (bin_z+0.5)*self.CENTER_BIN_SIZE),
        ])
        return center_cls, center_res

    def class2center(self, center_cls, center_res, point):
        # recover true center from center_cls and center_res
        bin_center = np.array([
            center_cls[0] * self.CENTER_BIN_SIZE + self.CENTER_BIN_SIZE/2 - self.CENTER_SEARCH_RANGE + point[0],
            point[1],
            center_cls[1] * self.CENTER_BIN_SIZE + self.CENTER_BIN_SIZE/2 - self.CENTER_SEARCH_RANGE + point[2]
        ])
        obj_center = bin_center + np.array([center_res[0]*self.CENTER_BIN_SIZE, center_res[1], center_res[2]*self.CENTER_BIN_SIZE])
        return obj_center

    def encode(self, obj, point):
        '''convert box3d related to a point to proposal vector'''
        # use point as origin
        obj_center = obj.t
        center_cls, center_res = self.center2class(obj_center, point)
        ## encode heading
        angle_cls, angle_res = self.angle2class(obj.ry)
        # print(angle_cls, angle_res)
        size_cls, size_res = self.size2class(np.array([obj.l,obj.w,obj.h]))
        return center_cls, center_res, angle_cls,angle_res, size_cls, size_res

    def tf_decode(self, end_points):
        """ Parse tensor output to 3d box
        Inputs:
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
        batch_size = end_points['center_x_scores'].shape[0]
        N = end_points['center_x_scores'].shape[1]
        # center
        bin_x = tf.argmax(end_points['center_x_scores'], axis=-1) # (B,N)
        bin_x_onehot = tf.one_hot(bin_x, depth=self.NUM_CENTER_BIN, axis=-1) # (B,N,NUM_CENTER_BIN)
        center_x_residuals_normalized = tf.reduce_sum(
            end_points['center_x_residuals_normalized']*tf.to_float(bin_x_onehot), axis=2) # BxN
        bin_z = tf.argmax(end_points['center_z_scores'], axis=-1) # (B,N)
        bin_z_onehot = tf.one_hot(bin_z, depth=self.NUM_CENTER_BIN, axis=-1) # (B,N,NUM_CENTER_BIN)
        center_z_residuals_normalized = tf.reduce_sum(
            end_points['center_z_residuals_normalized']*tf.to_float(bin_z_onehot), axis=2) # BxN
        center_y_residuals = end_points['center_y_residuals']
        # points_xyz = end_points['fg_points_xyz']
        bin_size = tf.constant(self.CENTER_BIN_SIZE, dtype=tf.float32)
        search_range = tf.constant(self.CENTER_SEARCH_RANGE, dtype=tf.float32)

        bin_center_y = tf.zeros([batch_size, N], tf.float32)
        bin_center = tf.stack([
            tf.to_float(bin_x) * bin_size + bin_size/2 - search_range,
            bin_center_y,
            tf.to_float(bin_z) * bin_size + bin_size/2 - search_range
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
        heading_one_hot = tf.one_hot(heading_cls, depth=self.NUM_HEADING_BIN, axis=-1)
        heading_res_norm = tf.reduce_sum(end_points['heading_residuals_normalized']*tf.to_float(heading_one_hot), axis=2) # BxN
        angle_per_class = tf.constant(2*self.HEADING_SEARCH_RANGE/float(self.NUM_HEADING_BIN), dtype=tf.float32)
        angle = tf.to_float(heading_cls) * angle_per_class + heading_res_norm * angle_per_class
        # to label format
        # angle = tf.where(angle > np.pi, angle - 2*np.pi, angle) # (B,N)
        angle = angle - self.HEADING_SEARCH_RANGE
        # size
        size_cls = tf.argmax(end_points['size_scores'], axis=-1)
        size_one_hot = tf.one_hot(size_cls, depth=NUM_SIZE_CLUSTER, axis=-1)
        size_one_hot_tiled = tf.tile(
            tf.expand_dims(tf.to_float(size_one_hot), -1), [1,1,1,3]) # BxNxNUM_SIZE_CLUSTERx3
        size_res_norm = tf.reduce_sum(
            end_points['size_residuals_normalized']*size_one_hot_tiled, axis=2) # BxNx3
        mean_size_arr_expand = tf.expand_dims(tf.expand_dims( \
            tf.constant(type_mean_size, dtype=tf.float32),0), 0) # NUM_SIZE_CLUSTERx3 -> 1x1xNUM_SIZE_CLUSTERx3
        mean_size_arr_expand_tiled = tf.tile(mean_size_arr_expand, [batch_size, N, 1, 1])
        mean_size = tf.reduce_sum( \
            size_one_hot_tiled * mean_size_arr_expand_tiled, axis=2) # BxNx3
        size_res = size_res_norm * mean_size
        box_size = mean_size + size_res # (B,N,3)

        return center, angle, box_size


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join('../', 'models'))
    from model_util import get_box3d_corners_helper
    class Box(object):
        pass
    obj = Box()
    obj.t = np.array([-28.3,1.2,5.2])
    obj.ry = -np.pi * 0.1
    obj.l = 1
    obj.w = 0.6
    obj.h = 0.7
    point = np.array([-27.1,1.3,4.3])
    NUM_HEADING_BIN = 9
    NUM_CENTER_BIN = 6
    CENTER_SEARCH_RANGE = 3.0
    HEADING_SEARCH_RANGE = 0.25*np.pi
    box_encoder = BoxEncoder(CENTER_SEARCH_RANGE, NUM_CENTER_BIN, HEADING_SEARCH_RANGE, NUM_HEADING_BIN)
    center_cls, center_res, angle_cls,angle_res, size_cls, size_res = box_encoder.encode(obj, point)

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
    fg_points_xyz = tf.constant(np.array([[point]]), dtype=tf.float32)
    centers, angles, sizes = box_encoder.tf_decode({
        # 'fg_points_xyz': tf.constant(np.array([[point]]), dtype=tf.float32),
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
    centers = centers + fg_points_xyz
    N = 1 * 1 # batch * num_point
    corners_3d = get_box3d_corners_helper(tf.reshape(centers, [N,3]), tf.reshape(angles, [N]), tf.reshape(sizes, [N,3]))
    with tf.Session() as sess:
        c, a, s = sess.run([centers, angles, sizes])
        print(obj.t, '<->', c[0][0])
        print(obj.ry, '<->', a[0][0])
        print([obj.l, obj.w, obj.h], '<->', s[0][0])
        corners_list = sess.run(corners_3d)
        print(corners_list)
