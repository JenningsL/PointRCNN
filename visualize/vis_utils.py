import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from obj_detection import obj_utils


def visualization(image_dir, index, flipped=False, display=True,
                  fig_size=(15, 9.15)):
    """Forms the plot figure and axis for the visualization

    Keyword arguments:
    :param image_dir -- directory of image files in the wavedata
    :param index -- index of the image file to present
    :param flipped -- flag to enable image flipping
    :param display -- display the image in non-blocking fashion
    :param fig_size -- (optional) size of the figure
    """

    def set_plot_limits(axes, image):
        # Set the plot limits to the size of the image, y is inverted
        axes.set_xlim(0, image.shape[1])
        axes.set_ylim(image.shape[0], 0)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True)
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.0)

    if not flipped:
        # Grab image data
        img = np.array(Image.open("%s/%06d.png" % (image_dir, index)),
                       dtype=np.uint8)
        # plot images
        ax1.imshow(img)
        ax2.imshow(img)

    else:
        # we want to flip the data so use cv2
        image_dir = "%s/%06d.png" % (image_dir, index)

        img = cv2.imread(image_dir)
        image_flipped = cv2.flip(img, 1)

        # plot images
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(image_flipped,
                                cv2.COLOR_BGR2RGB))

    set_plot_limits(ax1, img)
    set_plot_limits(ax2, img)

    if display:
        plt.show(block=False)

    return fig, ax1, ax2


def visualize_single_plot(image_dir, img_idx, is_video, flipped=False,
                          display=True, fig_size=(12.9, 3.9)):
    """Forms the plot figure and axis for the visualization

    Keyword arguments:
    :param image_dir -- directory of image files in the wavedata
    :param img_idx -- index of the image file to present
    :param flipped -- flag to enable image flipping
    :param display -- display the image in non-blocking fashion
    :param fig_size -- (optional) size of the figure
    """

    # Create the figure
    fig, ax = plt.subplots(1, figsize=fig_size, facecolor='black')
    fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0,
                        hspace=0.0, wspace=0.0)

    if not flipped:
        # Grab image data
        if is_video:
            img = np.array(Image.open("%s/%010d.png" % (image_dir, img_idx)),
                           dtype=np.uint8)
        else:
            img = np.array(Image.open("%s/%06d.png" % (image_dir, img_idx)),
                           dtype=np.uint8)
        # plot images
        ax.imshow(img)

    else:
        # we want to flip the data so use cv2
        if is_video:
            image_dir = "%s/%010d.png" % (image_dir, img_idx)
        else:
            image_dir = "%s/%06d.png" % (image_dir, img_idx)

        img = cv2.imread(image_dir)
        image_flipped = cv2.flip(img, 1)

        # plot images
        ax.imshow(cv2.cvtColor(image_flipped,
                               cv2.COLOR_BGR2RGB))

    # Set axes settings
    ax.set_axis_off()
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)

    # plot images
    ax.imshow(img)

    if display:
        plt.show(block=False)

    return fig, ax


def draw_box_2d(ax, obj, test_mode=False, color_tm='g'):
    """Draws the 2D boxes given the subplot and the object properties

    Keyword arguments:
    :param ax -- subplot handle
    :param obj -- object file to draw bounding box
    """

    if not test_mode:
        # define colors
        color_table = ["#00cc00", 'y', 'r', 'w']
        trun_style = ['solid', 'dashed']

        if obj.type != 'DontCare':
            # draw the boxes
            trc = int(obj.truncation > 0.1)
            rect = patches.Rectangle((obj.x1, obj.y1),
                                     obj.x2 - obj.x1,
                                     obj.y2 - obj.y1,
                                     linewidth=2,
                                     edgecolor=color_table[int(obj.occlusion)],
                                     linestyle=trun_style[trc],
                                     facecolor='none')

            # draw the labels
            label = "%s\n%1.1f rad" % (obj.type, obj.alpha)
            x = (obj.x1 + obj.x2) / 2
            y = obj.y1
            ax.text(x,
                    y,
                    label,
                    verticalalignment='bottom',
                    horizontalalignment='center',
                    color=color_table[int(obj.occlusion)],
                    fontsize=8,
                    backgroundcolor='k',
                    fontweight='bold')

        else:
            # Create a rectangle patch
            rect = patches.Rectangle((obj.x1, obj.y1),
                                     obj.x2 - obj.x1,
                                     obj.y2 - obj.y1,
                                     linewidth=2,
                                     edgecolor='c',
                                     facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    else:
        # we are in test mode, so customize the boxes differently
        # draw the boxes
        # we also don't care about labels here
        rect = patches.Rectangle((obj.x1, obj.y1),
                                 obj.x2 - obj.x1,
                                 obj.y2 - obj.y1,
                                 linewidth=2,
                                 edgecolor=color_tm,
                                 facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)


def draw_box_3d(ax, obj, p, show_orientation=True,
                color_table=None, line_width=3, double_line=True,
                box_color=None):
    """Draws the 3D boxes given the subplot, object label,
    and frame transformation matrix

    :param ax: subplot handle
    :param obj: object file to draw bounding box
    :param p:stereo frame transformation matrix

    :param show_orientation: optional, draw a line showing orientaion
    :param color_table: optional, a custom table for coloring the boxes,
        should have 4 values to match the 4 truncation values. This color
        scheme is used to display boxes colored based on difficulty.
    :param line_width: optional, custom line width to draw the box
    :param double_line: optional, overlays a thinner line inside the box lines
    :param box_color: optional, use a custom color for box (instead of
        the default color_table.
    """

    corners3d = obj_utils.compute_box_corners_3d(obj)
    corners, face_idx = obj_utils.project_box3d_to_image(corners3d, p)

    # define colors
    if color_table:
        if len(color_table) != 4:
            raise ValueError('Invalid color table length, must be 4')
    else:
        color_table = ["#00cc00", 'y', 'r', 'w']

    trun_style = ['solid', 'dashed']
    trc = int(obj.truncation > 0.1)

    if len(corners) > 0:
        for i in range(4):
            x = np.append(corners[0, face_idx[i, ]],
                          corners[0, face_idx[i, 0]])
            y = np.append(corners[1, face_idx[i, ]],
                          corners[1, face_idx[i, 0]])

            # Draw the boxes
            if box_color is None:
                box_color = color_table[int(obj.occlusion)]

            ax.plot(x, y, linewidth=line_width,
                    color=box_color,
                    linestyle=trun_style[trc])

            # Draw a thinner second line inside
            if double_line:
                ax.plot(x, y, linewidth=line_width / 3.0, color='b')

    if show_orientation:
        # Compute orientation 3D
        orientation = obj_utils.compute_orientation_3d(obj, p)

        if orientation is not None:
            x = np.append(orientation[0, ], orientation[0, ])
            y = np.append(orientation[1, ], orientation[1, ])

            # draw the boxes
            ax.plot(x, y, linewidth=4, color='w')
            ax.plot(x, y, linewidth=2, color='k')


def plot_3d_cube(corners, ax, c='lime'):
    """Plots 3D cube

    Arguments:
        corners: Bounding box corners
        ax: graphics handler
    """

    # Draw each line of the cube
    p1 = corners[:, 0]
    p2 = corners[:, 1]
    p3 = corners[:, 2]
    p4 = corners[:, 3]

    p5 = corners[:, 4]
    p6 = corners[:, 5]
    p7 = corners[:, 6]
    p8 = corners[:, 7]

    #############################
    # Bottom Face
    #############################
    ax.plot([p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            c=c)

    ax.plot([p2[0], p3[0]],
            [p2[1], p3[1]],
            [p2[2], p3[2]],
            c=c)

    ax.plot([p3[0], p4[0]],
            [p3[1], p4[1]],
            [p3[2], p4[2]],
            c=c)

    ax.plot([p4[0], p1[0]],
            [p4[1], p1[1]],
            [p4[2], p1[2]],
            c=c)

    #############################
    # Top Face
    #############################
    ax.plot([p5[0], p6[0]],
            [p5[1], p6[1]],
            [p5[2], p6[2]],
            c=c)

    ax.plot([p6[0], p7[0]],
            [p6[1], p7[1]],
            [p6[2], p7[2]],
            c=c)

    ax.plot([p7[0], p8[0]],
            [p7[1], p8[1]],
            [p7[2], p8[2]],
            c=c)

    ax.plot([p8[0], p5[0]],
            [p8[1], p5[1]],
            [p8[2], p5[2]],
            c=c)
    #############################
    # Front-Back Face
    #############################
    ax.plot([p5[0], p8[0]],
            [p5[1], p8[1]],
            [p5[2], p8[2]],
            c=c)

    ax.plot([p8[0], p4[0]],
            [p8[1], p4[1]],
            [p8[2], p4[2]],
            c=c)

    ax.plot([p4[0], p1[0]],
            [p4[1], p1[1]],
            [p4[2], p1[2]],
            c=c)

    ax.plot([p1[0], p5[0]],
            [p1[1], p5[1]],
            [p1[2], p5[2]],
            c=c)
    #############################
    # Front Face
    #############################
    ax.plot([p2[0], p3[0]],
            [p2[1], p3[1]],
            [p2[2], p3[2]],
            c=c)

    ax.plot([p3[0], p7[0]],
            [p3[1], p7[1]],
            [p3[2], p7[2]],
            c=c)

    ax.plot([p7[0], p6[0]],
            [p7[1], p6[1]],
            [p7[2], p6[2]],
            c=c)

    ax.plot([p6[0], p2[0]],
            [p6[1], p2[1]],
            [p6[2], p2[2]],
            c=c)


def cv2_show_image(window_name, image,
                   size_wh=None, location_xy=None):
    """ Helper function for specifying window size and location when
        displaying images with cv2

    :param window_name:
    :param image:
    :param size_wh:
    :param location_xy:
    """
    if size_wh is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, *size_wh)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    if location_xy is not None:
        cv2.moveWindow(window_name, *location_xy)

    cv2.imshow(window_name, image)
