from model_util import get_box3d_corners_helper

def project_to_image_tensor(points_3d, cam_p2_matrix):
    """Projects 3D points to 2D points in image space.

    Args:
        points_3d: a list of float32 tensor of shape [3, None]
        cam_p2_matrix: a float32 tensor of shape [3, 4] representing
            the camera matrix.

    Returns:
        points_2d: a list of float32 tensor of shape [2, None]
            This is the projected 3D points into 2D .i.e. corresponding
            3D points in image coordinates.
    """
    ones_column = tf.ones([1, tf.shape(points_3d)[1]])

    # Add extra column of ones
    points_3d_concat = tf.concat([points_3d, ones_column], axis=0)

    # Multiply camera matrix by the 3D points
    points_2d = tf.matmul(cam_p2_matrix, points_3d_concat)

    # 'Tensor' object does not support item assignment
    # so instead get the result of each division and stack
    # the results
    points_2d_c1 = points_2d[0, :] / points_2d[2, :]
    points_2d_c2 = points_2d[1, :] / points_2d[2, :]
    stacked_points_2d = tf.stack([points_2d_c1,
                                  points_2d_c2],
                                 axis=0)

    return stacked_points_2d

def tf_project_to_image_space(boxes, calib, image_shape):
    """
    Projects 3D tensor anchors into image space

    Args:
        boxes: a tensor of anchors in the shape [B, 7].
            The anchors are in the format [x, y, z, ry, h, w, l]
        calib: tensor [3, 4] stereo camera calibration p2 matrix
        image_shape: a float32 tensor of shape [2]. This is dimension of
            the image [h, w]

    Returns:
        box_corners: a float32 tensor corners in image space -
            N x [x1, y1, x2, y2]
        box_corners_norm: a float32 tensor corners as a percentage
            of the image size - N x [x1, y1, x2, y2]
    """
    box_center = tf.slice(boxes, [0,0], [-1, 3])
    box_angle = tf.slice(boxes, [0,3], [-1, 1])
    box_size = tf.slice(boxes, [0,4], [-1, 3])
    corners_3d = get_box3d_corners_helper(
        box_center, tf.gather(box_angle, 0, axis=-1), box_size)

    # Apply the 2D image plane transformation
    pts_2d = project_to_image_tensor(corners_3d, calib)

    # Get the min and maxes of image coordinates
    i_axis_min_points = tf.reduce_min(
        tf.reshape(pts_2d[0, :], (-1, 8)), axis=1)
    j_axis_min_points = tf.reduce_min(
        tf.reshape(pts_2d[1, :], (-1, 8)), axis=1)

    i_axis_max_points = tf.reduce_max(
        tf.reshape(pts_2d[0, :], (-1, 8)), axis=1)
    j_axis_max_points = tf.reduce_max(
        tf.reshape(pts_2d[1, :], (-1, 8)), axis=1)

    box_corners = tf.transpose(
        tf.stack(
            [i_axis_min_points, j_axis_min_points, i_axis_max_points,
             j_axis_max_points],
            axis=0))

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = tf.stack([image_shape_w, image_shape_h,
                                  image_shape_w, image_shape_h], axis=0)

    box_corners_norm = tf.divide(box_corners, image_shape_tiled)

    return box_corners, box_corners_norm
