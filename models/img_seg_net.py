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

    def get_seg_softmax(self):
        point_cloud = self.placeholders['pointclouds']
        mask_label = self.placeholders['seg_labels']
        end_points = self.end_points

        img_seg = self.graph.get_tensor_by_name('deeplab_v3/SemanticPredictions:0') # (B,360,1200,1)
        pts2d = projection.tf_rect_to_image(tf.slice(point_cloud,[0,0,0],[-1,-1,3]), self.placeholders['calib'])
        pts2d = tf.cast(pts2d, tf.int32) #(B,N,2)
        indices = tf.concat([
            tf.expand_dims(tf.tile(tf.range(0, self.batch_size), [self.num_point]), axis=-1), # (B*N, 1)
            tf.reshape(pts2d, [self.batch_size*self.num_point, 2])
        ], axis=-1) # (B*N,3)
        indices = tf.gather(indices, [0,2,1], axis=-1) # image's shape is (y,x)
        point_class = tf.reshape(
            tf.gather_nd(img_seg, indices), # (B*N,1)
            [self.batch_size, self.num_point, -1])  # (B,N,1)
        # person 11, rider 12, car 13
        point_class = tf.to_int32(tf.squeeze(point_class, axis=-1)) # (B,N)
        point_class = tf.to_int32(tf.equal(point_class, 13)) + tf.to_int32(tf.equal(point_class, 11)) * 2 + tf.to_int32(tf.equal(point_class, 12)) * 3
        img_seg_softmax = tf.one_hot(point_class, 4, axis=-1)
        return img_seg_softmax

if __name__ == '__main__':
    '''
    with open('img_seg.pkl', 'rb') as fin:
        preds = pickle.load(fin)
        labels = pickle.load(fin)
        print(np.sum(preds[0]==13))
        print(np.sum(labels[0]==1))

        tp = 0
        fp = 0
        fn = 0
        for i in range(100):
            pred = np.squeeze(preds[i], axis=-1)
            pred = (pred==13) + (pred==11)*2 + (pred==12)*3
            correct = np.sum(pred == labels[i])
            tp += np.sum(np.logical_and(pred == labels[i], labels[i] != 0))
            fp += np.sum(np.logical_and(pred != labels[i], labels[i] == 0))
            fn += np.sum(np.logical_and(pred != labels[i], labels[i] != 0))
        print('recall: {0}, precision: {1}'.format(float(tp)/(tp+fn), float(tp)/(tp+fp)))
    sys.exit()
    '''

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

