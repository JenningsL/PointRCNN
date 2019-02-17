import os
import unittest

import numpy as np

from wavedata.tools.obj_detection import obj_utils

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class ObjUtilsTest(unittest.TestCase):

    def test_is_point_inside(self):

        p1 = [1.0, 0.0, 0.0]
        p2 = [0.0, 0.0, 0.0]
        p3 = [0.0, 1.0, 0.0]
        p4 = [1.0, 1.0, 0.0]

        p5 = [1.0, 0.0, 1.0]
        p6 = [0.0, 0.0, 1.0]
        p7 = [0.0, 1.0, 1.0]
        p8 = [1.0, 1.0, 1.0]

        cube_corners = np.zeros((3, 8))
        cube_corners[0][0] = p1[0]
        cube_corners[0][1] = p2[0]
        cube_corners[0][2] = p3[0]
        cube_corners[0][3] = p4[0]
        cube_corners[0][4] = p5[0]
        cube_corners[0][5] = p6[0]
        cube_corners[0][6] = p7[0]
        cube_corners[0][7] = p8[0]

        cube_corners[1][0] = p1[1]
        cube_corners[1][1] = p2[1]
        cube_corners[1][2] = p3[1]
        cube_corners[1][3] = p4[1]
        cube_corners[1][4] = p5[1]
        cube_corners[1][5] = p6[1]
        cube_corners[1][6] = p7[1]
        cube_corners[1][7] = p8[1]

        cube_corners[2][0] = p1[2]
        cube_corners[2][1] = p2[2]
        cube_corners[2][2] = p3[2]
        cube_corners[2][3] = p4[2]
        cube_corners[2][4] = p5[2]
        cube_corners[2][5] = p6[2]
        cube_corners[2][6] = p7[2]
        cube_corners[2][7] = p8[2]

        # This point should lie within the cube
        x = [0.1, 0.2, 0.1]

        point_inside = obj_utils.is_point_inside(x, cube_corners)
        self.assertTrue(point_inside)

        # This should lie outside the cube
        y = [-0.1, 0.0, 0.0]

        point_inside = obj_utils.is_point_inside(y, cube_corners)
        self.assertFalse(point_inside)

    def test_get_point_filter(self):

        xz_plane = [0, -1, 0, 0]

        points = np.array([[0, 1, 0], [0, -1, 0], [5, 1, 5], [-5, 1, 5]])
        point_cloud = points.T

        # Test with offset planes at 0.5, and 2.0 distance
        filter1 = obj_utils.get_point_filter(
            point_cloud, [[-2, 2],
                          [-2, 2],
                          [-2, 2]],
            xz_plane, offset_dist=0.5)
        filter2 = obj_utils.get_point_filter(
            point_cloud, [[-2, 2],
                          [-2, 2],
                          [-2, 2]],
            xz_plane, offset_dist=2.0)

        self.assertEqual(np.sum(filter1), 1)
        self.assertEqual(np.sum(filter2), 2)

        filtered1 = points[filter1]
        filtered2 = points[filter2]

        self.assertEqual(len(filtered1), 1)
        self.assertEqual(len(filtered2), 2)

        np.testing.assert_allclose(filtered1, [[0, 1, 0]])
        np.testing.assert_allclose(filtered2, [[0, 1, 0], [0, -1, 0]])

    def test_object_label_eq(self):
        # Case 1, positive case
        object_1 = obj_utils.ObjectLabel()
        object_2 = obj_utils.ObjectLabel()
        self.assertTrue(object_1 == object_2)

        object_1.t = (1., 2., 3.)
        object_2.t = (1., 2., 3.)
        self.assertTrue(object_1 == object_2)

        # Case 2, negative case (single value)
        object_1 = {}  # Not a object label type
        object_2 = obj_utils.ObjectLabel()
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.truncation = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.occlusion = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.alpha = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.x1 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.y1 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.x2 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.y2 = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.h = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.w = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.l = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.t = (1., 1., 1.)
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.ry = 1.
        self.assertFalse(object_1 == object_2)

        object_1 = obj_utils.ObjectLabel()
        object_1.score = 1.
        self.assertFalse(object_1 == object_2)

        # Case 2, negative case (multiple values)
        object_1 = obj_utils.ObjectLabel()
        object_1.type = ""  # Type of object
        object_1.truncation = 1.
        object_1.occlusion = 1.
        object_1.alpha = 1.
        object_1.x1 = 1.
        object_1.y1 = 1.
        object_1.x2 = 1.
        object_1.y2 = 1.
        object_1.h = 1.
        object_1.w = 1.
        object_1.l = 1.
        object_1.t = [1., 1., 1.]
        object_1.ry = 1.
        object_1.score = 1.
        self.assertFalse(object_1 == object_2)


if __name__ == '__main__':
    unittest.main()
