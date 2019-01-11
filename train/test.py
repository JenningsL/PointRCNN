from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pickle
from threading import Thread
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from kitti import Dataset
from model_util import NUM_SEG_CLASSES, NUM_OBJ_CLASSES, g_type2onehotclass, NUM_CHANNEL

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
FLAGS = parser.parse_args()

# Set training configurations
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
# NUM_CHANNEL = 3 if FLAGS.no_intensity else 4 # point feature channel

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

def log_string(out_str):
    print(out_str)

TEST_DATASET = Dataset(NUM_POINT, '/data/ssd/public/jlliu/Kitti/object', 'val')

def test():
    # data loading threads
    test_produce_thread = Thread(target=TEST_DATASET.load, args=('/data/ssd/public/jlliu/PointRCNN/dataset/val',))
    test_produce_thread.start()

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, mask_labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)

            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Get model and losses
            end_points = {}
            end_points = MODEL.get_model(pointclouds_pl,
                is_training_pl, None, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, FLAGS.model_path)

        ops = {
            'pointclouds_pl': pointclouds_pl,
            'mask_labels_pl': mask_labels_pl,
            'is_training_pl': is_training_pl,
            'end_points': end_points}

    is_training = False
    log_string(str(datetime.now()))

    # To collect statistics
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_batches = 0

    pointclouds = []
    preds = []

    while(True):
        batch_pc, batch_mask_label, is_last_batch = TEST_DATASET.get_next_batch(1)

        feed_dict = {ops['pointclouds_pl']: batch_pc,
                     ops['mask_labels_pl']: batch_mask_label,
                     ops['is_training_pl']: is_training,}
        start = datetime.now()
        logits_val = sess.run([ops['end_points']['foreground_logits']], feed_dict=feed_dict)
        print('inference time: ', datetime.now() - start)
        #print(logits_val[0].shape)
        # segmentation acc
        preds_val = np.argmax(logits_val[0], 2)
        num_batches += 1
        # results
        pointclouds.append(batch_pc[0])
        preds.append(preds_val[0])
        # if is_last_batch:
        if num_batches >= 10:
            break

    with open('prediction.pkl','wb') as fp:
        pickle.dump(pointclouds, fp)
        pickle.dump(preds, fp)
    log_string('saved prediction')
    TEST_DATASET.stop_loading()
    test_produce_thread.join()

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    test()

