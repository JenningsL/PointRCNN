#!/usr/bin/env python3.5
import numpy as np
from PIL import Image, ImageDraw


def two_d_iou(box, boxes):
    """Compute 2D IOU between a 2D bounding box 'box' and a list

    :param box: a numpy array in the form of [x1, y1, x2, y2] where (x1,y1) are
    image coordinates of the top-left corner of the bounding box, and (x2,y2)
    are the image coordinates of the bottom-right corner of the bounding box.

    :param boxes: a numpy array formed as a list of boxes in the form
    [[x1, y1, x2, y2], [x1, y1, x2, y2]].

    :return iou: a numpy array containing 2D IOUs between box and every element
    in numpy array boxes.
    """
    iou = np.zeros(len(boxes), np.float64)

    x1_int = np.maximum(box[0], boxes[:, 0])
    y1_int = np.maximum(box[1], boxes[:, 1])
    x2_int = np.minimum(box[2], boxes[:, 2])
    y2_int = np.minimum(box[3], boxes[:, 3])

    w_int = x2_int - x1_int
    h_int = y2_int - y1_int

    non_empty = np.logical_and(w_int > 0, h_int > 0)

    if non_empty.any():
        intersection_area = np.multiply(w_int[non_empty], h_int[non_empty])

        box_area = (box[2] - box[0]) * (box[3] - box[1])

        boxes_area = np.multiply(
            boxes[non_empty, 2] - boxes[non_empty, 0],
            boxes[non_empty, 3] - boxes[non_empty, 1])

        union_area = box_area + boxes_area - intersection_area

        iou[non_empty] = intersection_area / union_area

    return iou.round(3)


def three_d_iou(box, boxes):
    """Computes approximate 3D IOU between a 3D bounding box 'box' and a list
    of 3D bounding boxes 'boxes'. All boxes are assumed to be aligned with
    respect to gravity. Boxes are allowed to rotate only around their z-axis.

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]
    :param boxes: a numpy array of the form:
        [[ry, l, h, w, tx, ty, tz], [ry, l, h, w, tx, ty, tz]]

    :return iou: a numpy array containing 3D IOUs between box and every element
        in numpy array boxes.
    """
    # First, rule out boxes that do not intersect by checking if the spheres
    # which inscribes them intersect.

    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    box_diag = np.sqrt(np.square(box[1]) +
                       np.square(box[2]) +
                       np.square(box[3])) / 2

    boxes_diag = np.sqrt(np.square(boxes[:, 1]) +
                         np.square(boxes[:, 2]) +
                         np.square(boxes[:, 3])) / 2

    dist = np.sqrt(np.square(boxes[:, 4] - box[4]) +
                   np.square(boxes[:, 5] - box[5]) +
                   np.square(boxes[:, 6] - box[6]))

    non_empty = box_diag + boxes_diag >= dist

    iou = np.zeros(len(boxes), np.float64)

    if non_empty.any():
        height_int, _ = height_metrics(box, boxes[non_empty])
        rect_int = get_rectangular_metrics(box, boxes[non_empty])

        intersection = np.multiply(height_int, rect_int)

        vol_box = np.prod(box[1:4])

        vol_boxes = np.prod(boxes[non_empty, 1:4], axis=1)

        union = vol_box + vol_boxes - intersection

        iou[non_empty] = intersection / union

    if iou.shape[0] == 1:
        iou = iou[0]

    return iou


def height_metrics(box, boxes):
    """Compute 3D height intersection and union between a box and a list of
    boxes

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],.....
                                        [ry, l, h, w, tx, ty, tz]]

    :return height_intersection: a numpy array containing the intersection along
    the gravity axis between the two bbs

    :return height_union: a numpy array containing the union along the gravity
    axis between the two bbs
    """
    boxes_heights = boxes[:, 2]
    boxes_centroid_heights = boxes[:, 5]

    min_y_boxes = boxes_centroid_heights - boxes_heights

    max_y_box = box[5]
    min_y_box = box[5] - box[2]

    max_of_mins = np.maximum(min_y_box, min_y_boxes)
    min_of_maxs = np.minimum(max_y_box, boxes_centroid_heights)

    offsets = min_of_maxs - max_of_mins
    height_intersection = np.maximum(0, offsets)

    height_union = np.maximum(min_y_box, boxes_centroid_heights) \
        - np.minimum(min_y_box, min_y_boxes) - \
        np.maximum(0, -offsets)

    return height_intersection, height_union


def get_rotated_3d_bb(boxes):
    """Compute rotated 3D bounding box coordinates.

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],...
                                         [ry, l, h, w, tx, ty, tz]]

    :return x: x coordinates of the four corners required to describe a 3D
    bounding box arranged as [[x1, x2, x3, x4],
                     [x1, x2, x3, x4],
                     ... ]

    :return z: z coordinates of the four corners required to describe a 3D
    bounding box arranged as [[z1, z2, z3, z4],
                     [z1, z2, z3, z4],
                     ... ].
    """

    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    x = np.array([[]])
    z = np.array([[]])

    for i in boxes:
        rot_mat = np.array([[np.cos(i[0]), np.sin(i[0])],
                            [-np.sin(i[0]), np.cos(i[0])]])

        x_corners = np.multiply(i[1] / 2, np.array([1, 1, -1, -1]))
        z_corners = np.multiply(i[3] / 2, np.array([1, -1, -1, 1]))

        temp_coor = np.dot(rot_mat, np.array([x_corners, z_corners]))

        # At the very first iteration, initialize x
        if x.shape[1] < 1:
            x = temp_coor[:1] + i[4]
            z = temp_coor[1:2] + i[6]
        # After that, append to the existing x
        else:
            x = np.append(x, temp_coor[:1] + i[4], axis=0)
            z = np.append(z, temp_coor[1:2] + i[6], axis=0)

    if x.shape[0] == 1:
        x = x[0]
        z = z[0]

    return x, z


def get_rectangular_metrics(box, boxes):
    """ Computes the intersection of the bases of oriented 3D bounding "box"
    and a set boxes of oriented 3D bounding boxes "boxes".

    :param box: a numpy array of the form: [ry, l, h, w, tx, ty, tz]

    :param boxes: a numpy array of the form: [[ry, l, h, w, tx, ty, tz],.....
                                        [ry, l, h, w, tx, ty, tz]]

    :return intersection: a numpy array containing intersection between the
    base of box and all other boxes.
    """
    if len(boxes.shape) == 1:
        boxes = np.array([boxes])

    mask_res = 0.01

    x_box, z_box = get_rotated_3d_bb(box)
    max_x_box = np.max(x_box)
    min_x_box = np.min(x_box)
    max_z_box = np.max(z_box)
    min_z_box = np.min(z_box)

    x_boxes, z_boxes = get_rotated_3d_bb(boxes)

    intersection = np.zeros(np.size(boxes, 0))

    if np.size(np.shape(x_boxes)) == 1:
        x_boxes = np.array([x_boxes])
        z_boxes = np.array([z_boxes])

    for i in range(np.size(boxes, 0)):
        x_i = x_boxes[i, :]
        z_i = z_boxes[i, :]
        test = max_x_box < np.min(x_i) or np.max(x_i) < min_x_box \
            or max_z_box < np.min(z_i) or np.max(z_i) < min_z_box

        if test:
            continue

        x_all = np.append(x_box, x_i)
        z_all = np.append(z_box, z_i)
        maxs = np.array([np.max(x_all), np.max(z_all)])
        mins = np.array([np.min(x_all), np.min(z_all)])

        mask_dims = np.int32(np.ceil((maxs - mins) / mask_res))

        mask_box_x = (x_box - mins[0]) / mask_res
        mask_box_z = (z_box - mins[1]) / mask_res
        mask_i_x = (x_i - mins[0]) / mask_res
        mask_i_z = (z_i - mins[1]) / mask_res
        # Drawing a binary image of the base of the two bounding boxes.
        # Then compute the element wise and of the two images to get the intersection.
        # Minor precision loss due to discretization.
        img = Image.new('L', (mask_dims[0], mask_dims[1]), 0)
        draw = ImageDraw.Draw(img, 'L')
        rect_coordinates = np.reshape(np.transpose(np.array([mask_box_x,
                                                             mask_box_z])), 8)
        rect_coordinates = np.append(rect_coordinates, rect_coordinates[0:2])
        draw.polygon(rect_coordinates.ravel().tolist(), outline=255, fill=255)
        del draw
        mask_box = np.asarray(img)

        img2 = Image.new('L', (mask_dims[0], mask_dims[1]), 0)
        draw = ImageDraw.Draw(img2, 'L')
        i_coordinates = np.reshape(np.transpose(np.array([mask_i_x,
                                                          mask_i_z])), 8)
        i_coordinates = np.append(i_coordinates, i_coordinates[0:2])
        draw.polygon(i_coordinates.ravel().tolist(), outline=255, fill=255)
        del draw
        mask_i = np.asarray(img2)

        mask_intersection = np.logical_and(mask_box, mask_i)
        intersection[i] = min(100, np.size(np.flatnonzero(
            mask_intersection)) * np.square(mask_res))

    if intersection.shape[0] == 1:
        intersection = intersection[0]

    return intersection
