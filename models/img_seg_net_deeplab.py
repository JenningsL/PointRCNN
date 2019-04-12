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
import tf_util
import projection
import time

NUM_HEADING_BIN = 12
NUM_CENTER_BIN = 12
CENTER_SEARCH_RANGE = 3.0
HEADING_SEARCH_RANGE = np.pi
NUM_CHANNEL = 4

class ImgSegNet(object):
    """docstring for RPN."""
    def __init__(self, batch_size, num_point, num_channel=4):
        self.batch_size = batch_size
        self.num_point = num_point
        self.num_channel = num_channel
        self.end_points = {}

    def get_placeholders(self):
        batch_size = self.batch_size
        num_point = self.num_point
        return {
            'pointclouds': tf.placeholder(tf.float32, shape=(batch_size, num_point, self.num_channel)),
            'img_inputs': self.graph.get_tensor_by_name('deeplab_v3/ImageTensor:0'),
            'calib': tf.placeholder(tf.float32, shape=(batch_size, 3, 4)),
            'seg_labels': tf.placeholder(tf.int32, shape=(batch_size, num_point))
        }

    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="deeplab_v3")
        self.graph = tf.get_default_graph()
        self.placeholders = self.get_placeholders()
        self.end_points['seg_softmax'] = self.get_seg_softmax()
        self.end_points['full_seg'] = self.get_semantic_seg()

    def get_semantic_seg(self):
        return self.graph.get_tensor_by_name('deeplab_v3/SemanticPredictions:0')

    def get_feature_map(self):
        # TODO: coarse feature map with size (batch_size, 100,350, 256)
        coarse_feature = self.graph.get_tensor_by_name('deeplab_v3/decoder/decoder_conv1_pointwise/Relu:0')
        return coarse_feature
        resized_feature = tf.image.resize_images(
            coarse_feature,
            [360,1200],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            align_corners=True)
        return resized_feature

    def get_seg_softmax(self):
        point_cloud = self.placeholders['pointclouds']
        mask_label = self.placeholders['seg_labels']
        end_points = self.end_points

        img_seg = self.graph.get_tensor_by_name('deeplab_v3/SemanticPredictions:0') # (B,360,1200,NUM_CLASSES)
        pts2d = projection.tf_rect_to_image(tf.slice(point_cloud,[0,0,0],[-1,-1,3]), self.placeholders['calib'])
        pts2d = tf.cast(pts2d, tf.int32) #(B,N,2)
        indices = tf.concat([
            tf.expand_dims(tf.tile(tf.range(0, self.batch_size), [self.num_point]), axis=-1), # (B*N, 1)
            tf.reshape(pts2d, [self.batch_size*self.num_point, 2])
        ], axis=-1) # (B*N,3)
        indices = tf.gather(indices, [0,2,1], axis=-1) # image's shape is (y,x)
        point_softmax = tf.reshape(
            tf.gather_nd(img_seg, indices), # (B*N,NUM_CLASSES)
            [self.batch_size, self.num_point, -1])  # (B,N,NUM_CLASSES)
        return point_softmax

if __name__ == '__main__':
    BATCH_SIZE = 1
    NUM_POINT = 16384
    with tf.Graph().as_default() as graph:
        with tf.device('/gpu:0'):
            seg_net = ImgSegNet(BATCH_SIZE, NUM_POINT)
            seg_net.load_graph(sys.argv[1])
            pts_softmax = seg_net.get_seg_softmax()
            semantic_seg = seg_net.get_semantic_seg()
            feat_map = seg_net.get_feature_map()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
    TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val', is_training=True)
    #TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'train', is_training=True)
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
        start = time.time()
        logits_val, full_output, feat_map_val = sess.run([pts_softmax, semantic_seg, feat_map], feed_dict=feed_dict)
        print('infer time:', time.time()-start)
        # save segmentation logits map
        for i in range(len(batch_data['ids'])):
            np.save(os.path.join('./rcnn_data', batch_data['ids'][i]+'_seg.npy'), full_output[i])
        # TODO: feature map is too large
        #print(feat_map_val.shape)
        #for i in range(len(batch_data['ids'])):
        #    np.save(os.path.join('./rcnn_data', batch_data['ids'][i]+'_feat.npy'), feat_map_val[i])
        # (batch_size, num_points, 1)
        preds_val = np.argmax(logits_val, axis=-1)
        '''
        max_val = np.amax(logits_val, axis=-1)
        max_val = np.squeeze(max_val, axis=-1)
        preds_val[max_val<0.99] = 0
        '''
        # NOTICE: batch_data['seg_label'] is (batch_size, num_points)
        #preds_val = np.squeeze(preds_val, axis=-1)
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

