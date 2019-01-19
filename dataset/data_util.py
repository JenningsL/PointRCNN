import numpy as np

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
