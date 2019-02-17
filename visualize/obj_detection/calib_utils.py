import csv
import numpy as np
import cv2
import os


class FrameCalibrationData:
    """Frame Calibration Holder
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters.

        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.

        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.tr_velodyne_to_cam = []


class StereoCalibrationData:
    """Stereo Calibration Holder
        1    baseline    distance between the two camera centers.

        1    f    focal length.

        3x3    k    intrinsic calibration matrix.

        3x4    p    camera matrix.

        1    center_u    camera origin u coordinate.

        1    center_v    camera origin v coordinate.
    """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.p = []
        self.center_u = 0.0
        self.center_v = 0.0


def read_calibration(calib_dir, img_idx):
    """Reads in Calibration file from Kitti Dataset.

    Keyword Arguments:
    ------------------
    calib_dir : Str
                Directory of the calibration files.

    img_idx : Int
              Index of the image.

    cam : Int
          Camera used from 0-3.

    Returns:
    --------
    frame_calibration_info : FrameCalibrationData
                             Contains a frame's full calibration data.

    """
    frame_calibration_info = FrameCalibrationData()

    data_file = open(calib_dir + "/%06d.txt" % img_idx, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calibration_info.p0 = p_all[0]
    frame_calibration_info.p1 = p_all[1]
    frame_calibration_info.p2 = p_all[2]
    frame_calibration_info.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calibration_info.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calibration_info.tr_velodyne_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calibration_info


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calibration_info = StereoCalibrationData()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calibration_info.baseline = abs(t_left[0] - t_right[0])
    stereo_calibration_info.f = k_left[0, 0]
    stereo_calibration_info.center_u = k_left[0, 2]
    stereo_calibration_info.center_v = k_left[1, 2]
    stereo_calibration_info.k = k_left
    stereo_calibration_info.p = left_cam_mat

    return stereo_calibration_info


def depth_from_disparity(disp, stereo_calibration_info, flatten_order='C'):
    """Transform disparity map to 3d point cloud.

    Camera coordinate frame:
    X: right
    Y: down
    Z: forward

    Example Usage found in:
        /demo/kitti

    Keyword Arguments:
    ------------------
    disp : cv2 mat
           disparity image.

    stereo_calibration_info : Instance of StereoCalibrationData class
                              Contains frame's stereo calibration info.

    flatten_order : (optional) see numpy.ndarray.flatten
        Specifies the way the depth array is flattened
        'C' - (default) row-major (C-style) order
        'F' - column-major (Fortran- style) order

    Returns:
    --------
    x : nd array
        x-coordinates of point cloud, every pixel has a value. Arranged in row
         major format.

    y : nd array
        y-coordinates of point cloud, every pixel has a value. Arranged in row
         major format

    z : nd array
        z-coordinates of point cloud, every pixel has a value. Arranged in row
         major format

      """

    disp = np.single(disp)
    disp = np.divide(disp, 256)
    disp[disp == 0] = 0.1

    depth = np.ones(disp.shape, np.single)
    depth = np.multiply(depth,
                        stereo_calibration_info.f *
                        stereo_calibration_info.baseline)

    depth = np.divide(depth, np.double(disp))

    sz = np.shape(depth)
    depth = depth.flatten(flatten_order)

    xx, yy = np.meshgrid(
        np.arange(1, sz[1] + 1, 1), np.arange(1, sz[0] + 1, 1))

    xx = xx.flatten(flatten_order) - stereo_calibration_info.center_u
    yy = yy.flatten(flatten_order) - stereo_calibration_info.center_v

    temp = np.divide(depth, stereo_calibration_info.f)

    x = np.multiply(xx, temp)
    y = np.multiply(yy, temp)
    z = depth

    return x, y, z


def project_to_image(point_cloud, p):
    """ Projects a 3D point cloud to 2D points for plotting

    :param point_cloud: 3D point cloud (3, N)
    :param p: Camera matrix (3, 4)

    :return: pts_2d: the image coordinates of the 3D points in the shape (2, N)
    """

    pts_2d = np.dot(p, np.append(point_cloud,
                                 np.ones((1, point_cloud.shape[1])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    pts_2d = np.delete(pts_2d, 2, 0)
    return pts_2d


def read_disparity(disp_dir, img_idx):
    """Reads in Disparity file from Kitti Dataset.

        Keyword Arguments:
        ------------------
        calib_dir : Str
                    Directory of the disparity files.

        img_idx : Int
                  Index of the image.

        Returns:
        --------
        disp_img : Numpy Array
                   Contains the disparity image.

        [] : if file is not found

        """
    disp_path = disp_dir + "/%06d_left_disparity.png" % img_idx

    if os.path.exists(disp_path):
        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
        return disp_img
    else:
        return []


def read_lidar(velo_dir, img_idx):
    """Reads in PointCloud from Kitti Dataset.

        Keyword Arguments:
        ------------------
        velo_dir : Str
                    Directory of the velodyne files.

        img_idx : Int
                  Index of the image.

        Returns:
        --------
        x : Numpy Array
                   Contains the x coordinates of the pointcloud.
        y : Numpy Array
                   Contains the y coordinates of the pointcloud.
        z : Numpy Array
                   Contains the z coordinates of the pointcloud.
        i : Numpy Array
                   Contains the intensity values of the pointcloud.

        [] : if file is not found

        """
    velo_dir = velo_dir + "/%06d.bin" % img_idx

    if os.path.exists(velo_dir):
        with open(velo_dir, 'rb') as fid:
            data_array = np.fromfile(fid, np.single)

        xyzi = data_array.reshape(-1, 4)

        x = xyzi[:, 0]
        y = xyzi[:, 1]
        z = xyzi[:, 2]
        i = xyzi[:, 3]

        return x, y, z, i
    else:
        return []


def lidar_to_cam_frame(xyz_lidar, frame_calib):
    """Transforms the pointclouds to the camera 0 frame.

        Keyword Arguments:
        ------------------
        xyz_lidar : N x 3 Numpy Array
                  Contains the x,y,z coordinates of the lidar pointcloud

        frame_calib : FrameCalibrationData
                  Contains calibration information for a given frame

        Returns:
        --------
        ret_xyz : Numpy Array
                   Contains the xyz coordinates of the transformed pointcloud.

        """

    # Pad the r0_rect matrix to a 4x4
    r0_rect_mat = frame_calib.r0_rect
    r0_rect_mat = np.pad(r0_rect_mat, ((0, 1), (0, 1)),
                         'constant', constant_values=0)
    r0_rect_mat[3, 3] = 1

    # Pad the tr_vel_to_cam matrix to a 4x4
    tf_mat = frame_calib.tr_velodyne_to_cam
    tf_mat = np.pad(tf_mat, ((0, 1), (0, 0)),
                    'constant', constant_values=0)
    tf_mat[3, 3] = 1

    # Pad the pointcloud with 1's for the transformation matrix multiplication
    one_pad = np.ones(xyz_lidar.shape[0]).reshape(-1, 1)
    xyz_lidar = np.append(xyz_lidar, one_pad, axis=1)

    # p_cam = P2 * R0_rect * Tr_velo_to_cam * p_velo
    rectified = np.dot(r0_rect_mat, tf_mat)
    ret_xyz = np.dot(rectified, xyz_lidar.T)

    # Change to N x 3 array for consistency.
    return ret_xyz[0:3].T
