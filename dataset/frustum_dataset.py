from __future__ import print_function

import os
import sys
import numpy as np
import copy
import random
import threading
import time
import cPickle as pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'kitti'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'visualize/mayavi'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_util import extract_pc_in_box3d, ProposalObject, random_shift_box3d
from kitti_object import *
import kitti_util as utils
from frustum_model_util import g_type2class, g_class2type, g_type2onehotclass, g_type_mean_size
from frustum_model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, REG_IOU
from frustum_model_util import type_whitelist
from provider import *
from shapely.geometry import Polygon, MultiPolygon
from Queue import Queue
from sklearn.neighbors import KDTree
from nms_rotate import nms_rotate_cpu

def is_near(prop1, prop2):
    c1 = np.array(prop1.t)
    c2 = np.array(prop2.t)
    r = max(prop1.w, prop1.l, prop1.h, prop2.w, prop2.l, prop2.h)
    return np.linalg.norm(c1-c2) < r / 2.0

def get_iou(bev_box1, bev_box2):
    p1 = Polygon(bev_box1)
    p2 = Polygon(bev_box2)
    intersection = p1.intersection(p2).area
    return intersection / (p1.area + p2.area - intersection)

class Sample(object):
    def __init__(self, idx, point_set, seg, box3d_center, angle_class, angle_residual,\
        size_class, size_residual, rot_angle, cls_label, proposal, heading_angle, iou, img_seg_map, calib):
        self.idx = idx
        self.heading_angle = heading_angle
        self.point_set = point_set
        self.seg_label = seg
        self.box3d_center = box3d_center
        self.angle_class = angle_class
        self.angle_residual = angle_residual
        self.size_class = size_class
        self.size_residual = size_residual
        self.rot_angle = rot_angle
        self.cls_label = cls_label
        self.img_seg_map = img_seg_map
        self.prop_box = np.array([proposal.t[0], proposal.t[1], proposal.t[2],
            proposal.l, proposal.h, proposal.w, proposal.ry])
        self.calib = calib
        # corresponding proposal without roi features
        prop_box = [proposal.t[0], proposal.t[1], proposal.t[2], proposal.l, proposal.h, proposal.w, proposal.ry]
        self.proposal = ProposalObject(prop_box, proposal.score, proposal.type)
        self.iou = iou

    def random_flip(self):
        if np.random.random()>0.5: # 50% chance flipping
            self.point_set[:,0] *= -1
            self.box3d_center[0] *= -1
            self.heading_angle = np.pi - self.heading_angle

        self.angle_class, self.angle_residual = angle2class(self.heading_angle,
            NUM_HEADING_BIN)

    def random_shift(self):
        box3d_center = self.box3d_center
        dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
        shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
        self.point_set[:,2] += shift
        self.box3d_center[2] += shift


class FrustumDataset(object):
    def __init__(self, npoints, kitti_path, batch_size, split, data_dir, is_training=False,
                 augmentX=1, random_shift=False, rotate_to_center=False, random_flip=False,
                 use_gt_prop=False):
        self.npoints = npoints
        self.random_shift = random_shift
        self.random_flip = random_flip
        self.rotate_to_center = rotate_to_center
        self.kitti_path = kitti_path
        self.data_dir = data_dir
        self.split = split
        self.use_gt_prop = use_gt_prop
        #self.num_channel = 7
        self.num_channel = 4
        self.is_training = is_training
        if split in ['train', 'val']:
            self.kitti_dataset = kitti_object(kitti_path, 'training')
            self.frame_ids = self.load_split_ids(split)
            random.shuffle(self.frame_ids)
        else:
            self.kitti_dataset = kitti_object_video(
                os.path.join(kitti_path, 'image_02/data'),
                os.path.join(kitti_path, 'velodyne_points/data'),
                kitti_path)
            self.frame_ids = map(lambda x: '{:06}'.format(x), range(self.kitti_dataset.num_samples))
        self.cur_batch = -1
        self.load_progress = 0
        self.batch_size = batch_size
        self.augmentX = augmentX

        self.sample_id_counter = -1 # as id for sample
        self.stop = False # stop loading thread
        self.last_sample_id = None

        self.sample_buffer = Queue(maxsize=1024)

    def get_proposals(self, rpn_out):
        proposals = []
        if self.split == 'train':
            nms_thres = 0.85
            max_keep = 300
        else:
            nms_thres = 0.8
            max_keep = 100
        bev_boxes = []
        for ry, center, size in zip(rpn_out['angles'], rpn_out['centers'], rpn_out['sizes']):
            bev_boxes.append([center[0], center[2], size[0], size[2], 180*ry/np.pi])
        bev_boxes = np.array(bev_boxes)
        nms_idx = nms_rotate_cpu(bev_boxes, rpn_out['scores'], nms_thres, max_keep)
        for ind in nms_idx:
            # to ProposalObject
            x,y,z = rpn_out['centers'][ind]
            l, h, w = rpn_out['sizes'][ind]
            ry = rpn_out['angles'][ind]
            proposal = ProposalObject(np.array([x,y,z,l, h, w, ry]))
            proposals.append(proposal)
        return proposals

    def load_split_ids(self, split):
        with open(os.path.join(self.kitti_path, split + '.txt')) as f:
            return [line.rstrip('\n') for line in f]

    def do_sampling(self, frame_data, pos_ratio=0.5):
        if not self.is_training:
            return frame_data['samples']
        samples = frame_data['samples']
        pos_idxs = frame_data['pos_idxs']
        neg_idxs = [i for i in range(0, len(samples)) if i not in pos_idxs]
        random.shuffle(neg_idxs)

        if pos_ratio == 0.0:
            keep_idxs = neg_idxs
        elif pos_ratio == 1.0:
            keep_idxs = pos_idxs
        else:
            '''
            cyclist_idxs = [i for i in pos_idxs if samples[i].cls_label == g_type2onehotclass['Cyclist']]
            pedestrian_idxs = [i for i in pos_idxs if samples[i].cls_label == g_type2onehotclass['Pedestrian']]
            car_idxs = [i for i in pos_idxs if samples[i].cls_label == g_type2onehotclass['Car']]
            # downsample
            car_idxs = random.sample(car_idxs, int(len(car_idxs) * 0.5))
            # oversample
            cyclist_idxs = cyclist_idxs * 10
            pedestrian_idxs = pedestrian_idxs * 5
            pos_idxs = car_idxs + cyclist_idxs + pedestrian_idxs
            '''
            need_neg = int(len(pos_idxs) * ((1-pos_ratio)/pos_ratio))
            keep_idxs = pos_idxs + neg_idxs[:need_neg]
        random.shuffle(keep_idxs)
        p = 0
        n = 0
        for i in keep_idxs:
            if samples[i].cls_label != g_type2onehotclass['NonObject']:
                p += 1
            else:
                n += 1
        kept_samples = [samples[i] for i in keep_idxs]

        # Data augmentation
        for sample in kept_samples:
            if self.random_flip:
                sample.random_flip()
            if self.random_shift:
                sample.random_shift()

        #print('Sampling result: pos {}, neg {}'.format(p, n))
        return kept_samples

    def stop_loading(self):
        self.stop = True
        while not self.sample_buffer.empty():
            item = self.sample_buffer.get()
            self.sample_buffer.task_done()

    def load(self, pos_ratio=0.5):
        i = -1
        last_sample_id = None
        while not self.stop:
            frame_id = self.frame_ids[i]
            frame_data = self.load_frame_data(frame_id)
            samples = self.do_sampling(frame_data, pos_ratio=pos_ratio)
            for s in samples:
                s.frame_id = frame_id
                self.sample_buffer.put(s)
            # update last_sample_id
            if len(samples) > 0:
                last_sample_id = samples[-1].idx
            # reach end
            if i == len(self.frame_ids) - 1:
                self.last_sample_id = last_sample_id
                random.shuffle(self.frame_ids)
            i = (i + 1) % len(self.frame_ids)

    def get_next_batch(self, wait=True):
        is_last_batch = False
        samples = []
        for _ in range(self.batch_size):
            if not wait and self.sample_buffer.empty(): # stop if empty, for inference
                is_last_batch = True
                break
            sample = self.sample_buffer.get() # block if empty, for training
            samples.append(sample)
            if sample.idx == self.last_sample_id:
                is_last_batch = True
                self.last_sample_id = None
                break

        batch_size = self.batch_size
        avail_num = len(samples) # note that avail_num can be smaller than self.batch_size
        # may pad the remaining with zero
        batch_data = {
            'ids': [],
            'pointcloud': np.zeros((batch_size, self.npoints, self.num_channel)),
            'cls_label': np.zeros((batch_size,), dtype=np.int32),
            'ious': np.zeros((batch_size,), dtype=np.float32),
            'seg_label': np.zeros((batch_size, self.npoints), dtype=np.int32),
            'center': np.zeros((batch_size, 3)),
            'heading_class': np.zeros((batch_size,), dtype=np.int32),
            'heading_residual': np.zeros((batch_size,)),
            'size_class': np.zeros((batch_size,), dtype=np.int32),
            'size_residual': np.zeros((batch_size, 3)),
            'rot_angle': np.zeros((batch_size,)),
            'img_seg_map': np.zeros((batch_size, 360, 1200, 4), dtype=np.float32),
            'prop_box': np.zeros((batch_size, 7), dtype=np.float32),
            'proposal_score': np.zeros((batch_size,), dtype=np.float32),
            'calib': np.zeros((batch_size, 3, 4), dtype=np.float32)
        }
        for i in range(avail_num):
            sample = samples[i]
            assert(sample.point_set.shape[0] == sample.seg_label.shape[0])
            if sample.point_set.shape[0] == 0:
                point_set = np.array([[0.0, 0.0, 0.0, 0.0]])
                seg_label = np.array([0])
            else:
                # Resample
                choice = np.random.choice(sample.point_set.shape[0], self.npoints, replace=True)
                point_set = sample.point_set[choice, 0:self.num_channel]
                seg_label = sample.seg_label[choice]
            batch_data['ids'].append(sample.frame_id)
            batch_data['pointcloud'][i,...] = point_set
            batch_data['cls_label'][i] = sample.cls_label
            batch_data['ious'][i] = sample.iou
            batch_data['seg_label'][i,:] = seg_label
            batch_data['center'][i,:] = sample.box3d_center
            batch_data['heading_class'][i] = sample.angle_class
            batch_data['heading_residual'][i] = sample.angle_residual
            batch_data['size_class'][i] = sample.size_class
            batch_data['size_residual'][i] = sample.size_residual
            batch_data['rot_angle'][i] = sample.rot_angle
            batch_data['img_seg_map'][i] = sample.img_seg_map
            batch_data['prop_box'][i] = sample.prop_box
            batch_data['proposal_score'][i] = sample.proposal.score
            batch_data['calib'][i] = sample.calib
        return batch_data, is_last_batch

    def get_center_view_rot_angle(self, frustum_angle):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi/2.0 + frustum_angle

    def get_center_view_point_set(self, points, rot_angle):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(points)
        return rotate_pc_along_y(point_set, rot_angle)

    def get_center_view_box3d_center(self, box3d, rot_angle):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (box3d[0,:] + box3d[6,:])/2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center,0), rot_angle).squeeze()

    def get_box3d_center(self, box3d):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (box3d[0,:] + box3d[6,:])/2.0
        return box3d_center

    def get_one_sample(self, proposal, pc_rect, image, calib, iou, gt_box_3d, gt_object, data_idx_str, img_seg_map):
        '''convert to frustum sample format'''
        prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(proposal, calib.P)
        if prop_corners_image_2d is None:
            print('skip proposal behind camera')
            return False
        # get points within proposal box
        # expand proposal
        proposal_expand = copy.deepcopy(proposal)
        proposal_expand.l += 0.5
        proposal_expand.w += 0.5
        proposal_expand.h += 0.5
        _, prop_corners_3d = utils.compute_box_3d(proposal_expand, calib.P)
        _,prop_inds = extract_pc_in_box3d(pc_rect, prop_corners_3d)
        pc_in_prop_box = pc_rect[prop_inds,:]
        seg_mask = np.zeros((pc_in_prop_box.shape[0]))
        if len(pc_in_prop_box) == 0:
            print('Reject proposal with no point')
            return False

        # Get frustum angle
        box2d_center = calib.project_rect_to_image(np.array([proposal.t]))[0]

        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
            box2d_center_rect[0,0])

        if gt_object is not None:
            obj_type = gt_object.type

            _,inds = extract_pc_in_box3d(pc_in_prop_box, gt_box_3d)
            seg_mask[inds] = 1
            # Reject object with too few point
            if np.sum(seg_mask) < 5:
                print('Reject object with too few point')
                return False

            # Get 3D BOX heading
            heading_angle = gt_object.ry
            # Get 3D BOX size
            box3d_size = np.array([gt_object.l, gt_object.w, gt_object.h])
        else:
            obj_type = 'NonObject'
            gt_box_3d = np.zeros((8, 3))
            heading_angle = 0
            box3d_size = np.zeros((1, 3))
            #frustum_angle = 0

        rot_angle = self.get_center_view_rot_angle(frustum_angle)

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(pc_in_prop_box, rot_angle)
        else:
            point_set = pc_in_prop_box

        # ------------------------------ LABELS ----------------------------
        # classification
        # assert(obj_type in ['Car', 'Pedestrian', 'Cyclist', 'NonObject'])
        assert(obj_type in type_whitelist)
        cls_label = g_type2onehotclass[obj_type]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(gt_box_3d, rot_angle)
        else:
            box3d_center = self.get_box3d_center(gt_box_3d)

        # Heading
        if self.rotate_to_center:
            heading_angle = heading_angle - rot_angle

        # Size
        size_class, size_residual = size2class(box3d_size, obj_type)

        angle_class, angle_residual = angle2class(heading_angle,
            NUM_HEADING_BIN)

        self.sample_id_counter += 1
        return Sample(self.sample_id_counter, point_set, seg_mask, box3d_center, angle_class, angle_residual,\
            size_class, size_residual, rot_angle, cls_label, proposal, heading_angle, iou, img_seg_map, calib.P)

    def visualize_one_sample(self, old_points, expand_points, gt_box_3d, prop_box_3d, box2d_center_rect):
        import mayavi.mlab as mlab
        from mayavi_utils import draw_lidar, draw_lidar_simple, draw_gt_boxes3d
        # fig = draw_lidar(pc_in_prop_box, pts_color=(1,1,1))
        fig = draw_lidar(expand_points[:, :3], pts_color=(1,1,1))
        fig = draw_lidar(old_points[:, :3], pts_color=(0,1,0), fig=fig)
        fig = draw_gt_boxes3d([gt_box_3d], fig, color=(1, 0, 0))
        fig = draw_gt_boxes3d([prop_box_3d], fig, draw_text=False, color=(1, 1, 1))
        # roi_feature_map
        # roi_features_size = 7 * 7 * 32
        # img_roi_features = prop.roi_features[0:roi_features_size].reshape((7, 7, -1))
        # bev_roi_features = prop.roi_features[roi_features_size:].reshape((7, 7, -1))
        # img_roi_features = np.amax(img_roi_features, axis=-1)
        # bev_roi_features = np.amax(bev_roi_features, axis=-1)
        # fig1 = mlab.figure(figure=None, bgcolor=(0,0,0),
        #     fgcolor=None, engine=None, size=(500, 500))
        # fig2 = mlab.figure(figure=None, bgcolor=(0,0,0),
        #     fgcolor=None, engine=None, size=(500, 500))
        # mlab.imshow(img_roi_features, colormap='gist_earth', name='img_roi_features', figure=fig1)
        # mlab.imshow(bev_roi_features, colormap='gist_earth', name='bev_roi_features', figure=fig2)
        mlab.plot3d([0, box2d_center_rect[0][0]], [0, box2d_center_rect[0][1]], [0, box2d_center_rect[0][2]], color=(1,1,1), tube_radius=None, figure=fig)
        raw_input()

    def get_proposals_from_label(self, labels, calib, augmentX):
        '''construct proposal from label'''
        proposals = []
        for label in labels:
            for _ in range(augmentX):
                prop = ProposalObject(list(label.t) + [label.l, label.h, label.w, label.ry], 1.0, label.type, None)
                prop = random_shift_box3d(prop)
                proposals.append(prop)

        return proposals

    def visualize_proposals(self, pc_rect, prop_boxes, neg_boxes, gt_boxes, pc_seg=None):
        import mayavi.mlab as mlab
        from mayavi_utils import draw_lidar, draw_gt_boxes3d
        fig = draw_lidar(pc_rect)
        if pc_seg:
            fig = draw_lidar(pc_rect[pc_seg==1], fig=fig, pts_color=(1, 1, 1))
        fig = draw_gt_boxes3d(prop_boxes, fig, draw_text=False, color=(1, 0, 0))
        fig = draw_gt_boxes3d(neg_boxes, fig, draw_text=False, color=(0, 1, 0))
        fig = draw_gt_boxes3d(gt_boxes, fig, draw_text=False, color=(1, 1, 1))
        raw_input()

    def load_frame_data(self, data_idx_str, rpn_out=None, img_seg_map=None):
        data_idx = int(data_idx_str)
        # rpn out and img_seg_map can be directly provided
        if rpn_out is None or img_seg_map is None:
            try:
                with open(os.path.join(self.data_dir, data_idx_str+'.pkl'), 'rb') as fin:
                    rpn_out = pickle.load(fin)
                # load image segmentation output
                img_seg_map = np.load(os.path.join(self.data_dir, data_idx_str+'_seg.npy'))
            except Exception as e:
                print(e)
                return {'samples': [], 'pos_idxs': []}
        start = time.time()
        calib = self.kitti_dataset.get_calibration(data_idx) # 3 by 4 matrix
        image = self.kitti_dataset.get_image(data_idx)
        pc_velo = self.kitti_dataset.get_lidar(data_idx)
        img_height, img_width = image.shape[0:2]
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)
        pc_velo = pc_velo[img_fov_inds, :]
        # Same point sampling as RPN
        #choice = rpn_out['pc_choices']
        # print('choice', len(choice))
        #pc_velo = pc_velo[choice, :]

        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        gt_boxes_xy = []
        gt_boxes_3d = []
        if self.is_training:
            objects = self.kitti_dataset.get_label_objects(data_idx)
        else:
            objects = []
        objects = filter(lambda obj: obj.type in type_whitelist, objects)
        for obj in objects:
            _, gt_corners_3d = utils.compute_box_3d(obj, calib.P)
            gt_boxes_xy.append(gt_corners_3d[:4, [0,2]])
            gt_boxes_3d.append(gt_corners_3d)
        recall = np.zeros((len(objects),))

        if self.use_gt_prop:
            assert(self.is_training==True)
            proposals = self.get_proposals_from_label(objects, calib, self.augmentX)
        else:
            proposals = self.get_proposals(rpn_out)
            # add more training samples
            #if self.split == 'train':
            #    proposals += self.get_proposals_from_label(objects, calib, 1)

        samples = []
        pos_idxs = []
        pos_box = []
        neg_box = []
        avg_iou = []
        for prop_ in proposals:
            prop = copy.deepcopy(prop_)
            prop_corners_image_2d, prop_corners_3d = utils.compute_box_3d(prop_, calib.P)
            if prop_corners_image_2d is None:
                # print('skip proposal behind camera')
                continue
            # testing
            if not self.is_training:
                sample = self.get_one_sample(prop, pc_rect, image, calib, -1, None, None, data_idx_str, img_seg_map)
                if sample:
                    samples.append(sample)
                continue
            # training
            prop_box_xy = prop_corners_3d[:4, [0,2]]
            # find corresponding label object
            obj_idx, iou_with_gt = self.find_match_label(prop_box_xy, gt_boxes_xy)

            # iou < 0.3 is no object, iou >= 0.5 is object, iou > 0.65 will be used to
            # train regression
            if iou_with_gt < 0.3:
                # non-object
                sample = self.get_one_sample(prop, pc_rect, image, calib, iou_with_gt, None, None, data_idx_str, img_seg_map)
                if sample:
                    samples.append(sample)
                    neg_box.append(prop_corners_3d)
            elif iou_with_gt >= 0.6 \
                or (iou_with_gt >= 0.5 and objects[obj_idx].type in ['Pedestrian', 'Cyclist']):
                obj_type = objects[obj_idx].type
                avg_iou.append(iou_with_gt)

                sample = self.get_one_sample(prop, pc_rect, image, calib, iou_with_gt, gt_boxes_3d[obj_idx], objects[obj_idx], data_idx_str, img_seg_map)
                if sample:
                    pos_idxs.append(len(samples))
                    samples.append(sample)
                    recall[obj_idx] = 1
                    #_, prop_corners_3d = utils.compute_box_3d(prop, calib.P)
                    pos_box.append(prop_corners_3d)
            else:
                continue

        # self.visualize_proposals(pc_rect, pos_box, neg_box, gt_boxes_3d)
        self.load_progress += 1
        # print('load {} samples, pos {}'.format(len(samples), len(pos_idxs)))
        ret = {'samples': samples, 'pos_idxs': pos_idxs}
        if len(objects) > 0:
            ret['recall'] = np.sum(recall)/len(objects)
        if len(pos_idxs) > 0:
            ret['avg_iou'] = avg_iou
        #print('load frame data cost: ', time.time() - start)
        return ret

    def find_match_label(self, prop_corners, labels_corners):
        '''
        Find label with largest IOU. Label boxes can be rotated in xy plane
        '''
        # labels = MultiPolygon(labels_corners)
        labels = map(lambda corners: Polygon(corners), labels_corners)
        target = Polygon(prop_corners)
        largest_iou = 0
        largest_idx = -1
        for i, label in enumerate(labels):
            area1 = label.area
            area2 = target.area
            intersection = target.intersection(label).area
            iou = intersection / (area1 + area2 - intersection)
            # if a proposal cover enough ground truth, take it as positive
            #if intersection / area1 >= 0.8:
            #    iou = 0.66
            # print(area1, area2, intersection)
            # print(iou)
            if iou > largest_iou:
                largest_iou = iou
                largest_idx = i
        return largest_idx, largest_iou

if __name__ == '__main__':
    kitti_path = sys.argv[1]
    split = sys.argv[2]
    if split == 'train':
        augmentX = 5
        use_gt_prop = True
    else:
        augmentX = 1
        use_gt_prop = False
    dataset = FrustumDataset(512, kitti_path, 16, split, data_dir='./rcnn_data_'+split,
                 augmentX=augmentX, random_shift=True, rotate_to_center=True, random_flip=True,
                 use_gt_prop=use_gt_prop)
    #dataset.load(0.5)
    dataset.load_frame_data('000001')
    dataset.get_next_batch(wait=False)

    '''
    produce_thread = threading.Thread(target=dataset.load, args=(1.0,))
    produce_thread.start()

    while(True):
        batch = dataset.get_next_batch()
        is_last_batch = batch[-1]
        print(batch[0][0][0])
        break
        if is_last_batch:
            break
    dataset.stop_loading()

    produce_thread.join()
    '''
