import tensorflow as tf
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
    batch_size = boxes.shape[0]
    box_center = tf.slice(boxes, [0,0], [-1, 3])
    box_angle = tf.slice(boxes, [0,3], [-1, 1])
    box_size = tf.slice(boxes, [0,4], [-1, 3])
    corners_3d = get_box3d_corners_helper(
        box_center, tf.gather(box_angle, 0, axis=-1), box_size) # (B,8,3)
    #corners_3d_list = tf.reshape(corners_3d, [batch_size*8, 3])
    corners_3d = tf.expand_dims(corners_3d, axis=2) # (B,8,1,3)
    calib_tiled = tf.tile(tf.expand_dims(calib, 1), [1,8,1,1]) # (B,8,3,4)
    projected_pts = tf.matmul(corners_3d, calib_tiled) # (B,8,1,4)

    projected_pts_norm = projected_pts/tf.slice(projected_pts, [0,0,0,2], [-1,-1,-1,1]) # divided by depth

    corners_2d = tf.gather(tf.squeeze(projected_pts_norm, axis=2), [0,1], axis=-1) # (B,8,2)

    pts_2d_min = tf.reduce_min(corners_2d, axis=1)
    pts_2d_max = tf.reduce_max(corners_2d, axis=1) # (B, 2)
    box_corners = tf.stack([
        tf.gather(pts_2d_min, 0, axis=1),
        tf.gather(pts_2d_min, 1, axis=1),
        tf.gather(pts_2d_max, 0, axis=1),
        tf.gather(pts_2d_max, 1, axis=1),
        ], axis=1) # (B,4)

    # Normalize
    image_shape_h = image_shape[0]
    image_shape_w = image_shape[1]

    image_shape_tiled = tf.tile([[image_shape_w, image_shape_h, image_shape_w, image_shape_h]], [batch_size,1])

    box_corners_norm = box_corners / tf.to_float(image_shape_tiled)

    return box_corners, box_corners_norm
