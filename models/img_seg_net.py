from __future__ import print_function

import sys
import os
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from threading import Thread
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from rpn_dataset import Dataset
from rpn import NUM_SEG_CLASSES
import tf_util
import projection
from model_util import focal_loss
from img_vgg_pyramid import ImgVggPyr
from collections import namedtuple

class ImgSegNet(object):
    """docstring for ImgSegNet."""
    def __init__(self, batch_size, num_point, num_channel=4, is_training=True):
        self.batch_size = batch_size
        self.num_point = num_point
        self.num_channel = num_channel
        self.is_training = is_training
        self.end_points = {}
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
            'is_training_pl': tf.placeholder(tf.bool, shape=())
        }

    def build(self):
        point_cloud = self.placeholders['pointclouds']
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
        self.seg_logits = slim.conv2d(
            self.img_feature_maps,
            NUM_SEG_CLASSES, [1, 1],
            scope='bottleneck',
            normalizer_fn=slim.batch_norm,
            #normalizer_fn=None,
            normalizer_params={
                'is_training': self.is_training})

        pts2d = projection.tf_rect_to_image(tf.slice(point_cloud,[0,0,0],[-1,-1,3]), self.placeholders['calib'])
        pts2d = tf.cast(pts2d, tf.int32) #(B,N,2)
        indices = tf.concat([
            tf.expand_dims(tf.tile(tf.range(0, self.batch_size), [self.num_point]), axis=-1), # (B*N, 1)
            tf.reshape(pts2d, [self.batch_size*self.num_point, 2])
        ], axis=-1) # (B*N,3)
        indices = tf.gather(indices, [0,2,1], axis=-1) # image's shape is (y,x)
        self.end_points['foreground_logits'] = tf.reshape(
            tf.gather_nd(self.seg_logits, indices), # (B*N,C)
            [self.batch_size, self.num_point, -1])  # (B,N,C)

    def get_seg_softmax(self):
        img_seg_softmax = tf.nn.softmax(self.end_points['point_seg_logits'], axis=-1)
        return img_seg_softmax

    def get_loss(self):
        pls = self.placeholders
        end_points = self.end_points
        batch_size = self.batch_size
        # 3D Segmentation loss
        mask_loss = focal_loss(end_points['foreground_logits'], tf.one_hot(pls['seg_labels'], NUM_SEG_CLASSES, axis=-1))
        tf.summary.scalar('mask loss', mask_loss)
        return mask_loss

if __name__ == '__main__':
    BATCH_SIZE = 1
    NUM_POINT = 16384
    with tf.Graph().as_default() as graph:
        with tf.device('/gpu:0'):
            seg_net = ImgSegNet(BATCH_SIZE, NUM_POINT)
            seg_net.load_graph(sys.argv[1])
            seg_softmax = seg_net.get_seg_softmax()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
    TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val', is_training=True)
    test_produce_thread = Thread(target=TEST_DATASET.load, args=(False,))
    test_produce_thread.start()
    pls = seg_net.placeholders

    n = 0
    total_seen = 0
    tp = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
    fp = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
    fn = {'Car': 0, 'Pedestrian': 0, 'Cyclist': 0}
    g_type2onehotclass = {'NonObject': 0, 'Car': 1, 'Pedestrian': 2, 'Cyclist': 3}
    while(True):
        batch_data, is_last_batch = TEST_DATASET.get_next_batch(BATCH_SIZE)
        feed_dict = {
            pls['pointclouds']: batch_data['pointcloud'],
            pls['img_inputs']: batch_data['images'],
            pls['calib']: batch_data['calib'],
            pls['seg_labels']: batch_data['seg_label']
        }
        logits_val = sess.run(seg_softmax, feed_dict=feed_dict)
        preds_val = np.argmax(logits_val, axis=-1)
        if n ==0:
            print(preds_val.shape)
            print(preds_val[0])
        print('pred: {0}, label: {1}'.format(np.sum(preds_val==1), np.sum(batch_data['seg_label']==1)))
        correct = np.sum(preds_val == batch_data['seg_label'])
        for c in ['Car', 'Pedestrian', 'Cyclist']:
            one_hot_class = g_type2onehotclass[c]
            tp[c] += np.sum(np.logical_and(preds_val == batch_data['seg_label'], batch_data['seg_label'] == one_hot_class))
            fp[c] += np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] != one_hot_class))
            fn[c] += np.sum(np.logical_and(preds_val != batch_data['seg_label'], batch_data['seg_label'] == one_hot_class))
        total_seen += NUM_POINT * BATCH_SIZE
        n += 1
        if is_last_batch:
            break
    print(tp, fp, fn)
    for c in ['Car', 'Pedestrian', 'Cyclist']:
        if (tp[c]+fn[c] == 0) or (tp[c]+fp[c]) == 0:
            continue
        print(c + ' segmentation recall: %f'% \
            (float(tp[c])/(tp[c]+fn[c])))
        print(c + ' segmentation precision: %f'% \
            (float(tp[c])/(tp[c]+fp[c])))
    TEST_DATASET.stop_loading()
    test_produce_thread.join()
