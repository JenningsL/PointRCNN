import sys
import os
import re
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import matplotlib.patheffects as patheffects

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'mayavi'))
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
import kitti_util as utils
from kitti_object import *
import mayavi.mlab as mlab
from mayavi_utils import draw_lidar, draw_gt_boxes3d

import vis_utils

type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
BOX_COLOUR_SCHEME = {
    'Car': '#00FF00',           # Green
    'Pedestrian': '#00FFFF',    # Teal
    'Cyclist': '#FFFF00'        # Yellow
}

class DetectObject(object):
    def __init__(self, h,w,l,tx,ty,tz,ry, frame_id, type_label, score, box_2d=None, box_3d=None):
        self.t = [tx,ty,tz]
        self.ry = ry
        self.h = h
        self.w = w
        self.l = l
        self.frame_id = frame_id
        self.type_label = type_label
        self.score = score
        self.box_2d = box_2d
        self.box_3d = box_3d # corners

def load_result(dataset, fname, data_idx):
    objs = []
    # calib = dataset.get_calibration(int(data_idx))
    with open(fname, 'r') as fin:
        for line in fin:
            cols = line.split()
            type_label = cols[0]
            h,w,l = (float(cols[8]), float(cols[9]), float(cols[10]))
            tx,ty,tz = (float(cols[11]), float(cols[12]), float(cols[13]))
            ry = float(cols[14])
            score = np.exp(float(cols[15]))
            obj = DetectObject(h,w,l,tx,ty,tz,ry,data_idx,type_label,score)
            objs.append(obj)
    return objs

def draw_boxes(objects, calib, plot_axes):
    all_corners = []
    for obj in objects:
        if hasattr(obj, 'type_label'):
            obj.obj_type = obj.type_label
        else:
            obj.obj_type = obj.type
        if not hasattr(obj, 'truncation'):
            obj.truncation = 0
        if not hasattr(obj, 'occlusion'):
            obj.occlusion = 0
        if not hasattr(obj, 'score'):
            obj.score = 1
        if obj.obj_type not in type_whitelist:
            continue
        vis_utils.draw_box_3d(plot_axes, obj, calib.P,
                          show_orientation=False,
                          color_table=['r', 'y', 'r', 'w'],
                          line_width=2,
                          double_line=False)
        box3d_pts_2d, corners = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            continue
        all_corners.append(corners)
        # draw text info
        x1 = np.amin(box3d_pts_2d, axis=0)[0]
        y1 = np.amin(box3d_pts_2d, axis=0)[1]
        x2 = np.amax(box3d_pts_2d, axis=0)[0]
        y2 = np.amax(box3d_pts_2d, axis=0)[1]
        text_x = (x1 + x2) / 2
        text_y = y1
        text = "{}\n{:.2f}".format(obj.obj_type, obj.score)
        plot_axes.text(text_x, text_y - 4,
            text,
            verticalalignment='bottom',
            horizontalalignment='center',
            color=BOX_COLOUR_SCHEME[obj.obj_type],
            fontsize=10,
            fontweight='bold',
            path_effects=[
                patheffects.withStroke(linewidth=2,
                                       foreground='black')])
    return all_corners

def visualize(dataset, frame_id, prediction, show_3d=False, output_dir=None):
    fig_size = (10, 6.1)
    is_video = type(dataset).__name__ == 'kitti_object_video'
    # pred_fig, pred_2d_axes, pred_3d_axes = \
    #     vis_utils.visualization(dataset.image_dir,
    #                             int(frame_id),
    #                             display=False,
    #                             fig_size=fig_size)
    pred_fig, pred_3d_axes = vis_utils.visualize_single_plot(
        dataset.image_dir, int(frame_id), is_video, flipped=False,
        display=False, fig_size=fig_size)
    calib = dataset.get_calibration(frame_id) # 3 by 4 matrix

    # 2d visualization
    # draw groundtruth
    # labels = dataset.get_label_objects(frame_id)
    # draw_boxes(labels, calib, pred_2d_axes)
    # draw prediction on second image
    pred_corners = draw_boxes(prediction, calib, pred_3d_axes)
    if output_dir:
        filename = os.path.join(output_dir, 'result_2d_image/%06d.png' % frame_id)
        plt.savefig(filename)
        plt.close(pred_fig)
    else:
        plt.show()
        #input()

    if show_3d:
        # 3d visualization
        pc_velo = dataset.get_lidar(frame_id)
        boxes3d_velo = []
        for corners in pred_corners:
            pts_velo = calib.project_rect_to_velo(corners)
            boxes3d_velo.append(pts_velo)
        fig = draw_lidar(pc_velo)
        fig = draw_gt_boxes3d(boxes3d_velo, fig, draw_text=False, color=(1, 1, 1))
        #input()
        if output_dir:
            filename = os.path.join(output_dir, 'result_3d_image/%06d.png' % frame_id)
            mlab.savefig(filename, figure=fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_path',
                    type=str,
                    dest='kitti_path',
                    required=True,
                    help='kitti_path')
    parser.add_argument('--kitti_type',
                    type=str,
                    dest='kitti_type',
                    required=True,
                    help='object or video')
    parser.add_argument('--detection_path',
                    type=str,
                    dest='detection_path',
                    required=True,
                    help='detection_path')
    parser.add_argument('--output_dir',
                    type=str,
                    dest='output_dir',
                    help='output_dir')

    args = parser.parse_args()

    if args.kitti_type == 'object':
        dataset = kitti_object(args.kitti_path, 'training')
    else:
        dataset = kitti_object_video(
            os.path.join(args.kitti_path, 'image_02/data'),
            os.path.join(args.kitti_path, 'velodyne_points/data'),
            args.kitti_path)

    if args.output_dir:
        if not os.path.exists(os.path.join(args.output_dir, 'result_2d_image')):
            os.makedirs(os.path.join(args.output_dir, 'result_2d_image'))
        if not os.path.exists(os.path.join(args.output_dir, 'result_3d_image')):
            os.makedirs(os.path.join(args.output_dir, 'result_3d_image'))

    for f in os.listdir(args.detection_path):
        print('processing %s' % f)
        data_idx = f.replace('.txt', '')
        fname = os.path.join(args.detection_path, f)
        objs = load_result(dataset, fname, data_idx)

        visualize(dataset, int(data_idx), objs, show_3d=False, output_dir=args.output_dir)
