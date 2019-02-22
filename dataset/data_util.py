import numpy as np
from shapely.geometry import Polygon

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotate_points_along_y(points, angle):
    R = roty(angle)
    return np.dot(points, R)

def shift_point_cloud(points, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, shifted point clouds
    """
    N, C = points.shape
    shifts = np.random.uniform(-shift_range, shift_range, (N,3))
    return points + shifts

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def np_read_lines(filename, lines):
    arr = []
    with open(filename, 'rb') as fp:
        for i, line in enumerate(fp):
            if i in lines:
                arr.append(np.fromstring(line, dtype=float, sep=' '))
    return np.array(arr)

class ProposalObject(object):
    def __init__(self, box_3d, score=0.0, type='Car', roi_features=None):
        # [x, y, z, l, h, w, ry]
        self.t = box_3d[0:3]
        self.l = box_3d[3]
        self.h = box_3d[4]
        self.w = box_3d[5]
        self.ry = box_3d[6]
        self.score = score
        self.type = type
        self.roi_features = roi_features

def find_match_label(prop_corners, labels_corners):
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
        if intersection / area1 >= 0.8:
           iou = 0.66
        # print(area1, area2, intersection)
        # print(iou)
        if iou > largest_iou:
            largest_iou = iou
            largest_idx = i
    return largest_idx, largest_iou

def random_shift_box3d(obj, shift_ratio=0.1):
    '''
    Randomly w, l, h
    '''
    r = shift_ratio
    # 0.9 to 1.1
    obj.t[0] = obj.t[0] + obj.l*r*(np.random.random()*2-1)
    obj.t[1] = obj.t[1] + obj.w*r*(np.random.random()*2-1)
    obj.w = obj.w*(1+np.random.random()*2*r-r)
    obj.l = obj.l*(1+np.random.random()*2*r-r)
    obj.ry += (np.pi / 3.6 * (np.random.random()*2*r-r)) # -5~5 degree
    if obj.ry > np.pi:
        obj.ry -= 2*np.pi
    elif obj.ry < -np.pi:
        obj.ry += 2*np.pi
    # obj.h = obj.h*(1+np.random.random()*2*r-r)
    return obj
