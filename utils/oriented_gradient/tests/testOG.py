from unittest import TestCase

import cv2
import numpy as np
from scipy import ndimage

from utils.oriented_gradient.OrientedGradientCalculator import OrientedGradientCalculator


class TestOG(TestCase):
    def setUp(self):
        self.img = cv2.imread('gaussian_test_img.png')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.img = self.img[:, :, 0]

    def test_calculate_oriented_gradient(self):
        object_under_test = OrientedGradientCalculator(self.img, 3, 0)
        result = object_under_test.calculate()
        np.testing.assert_array_almost_equal(np.full((result.shape[0], result.shape[1]), 0.), result)
        print("Done")

    def test_calculate_oriented_gradient2(self):
        object_under_test = OrientedGradientCalculator(self.img, 2, 0)
        result = object_under_test.calculate()
        result = np.around(result, 2)
        np.testing.assert_array_equal(result[3:8, 3:8],
                                      np.array([[0.67, 2., 2., 2., 0.67],
                                                [0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0.],
                                                [0.67, 2., 2., 2., 0.67]]))
        print("Done")

    def test_calculate_oriented_gradient_vis(self):
        channel = 2
        img = cv2.imread('orig.png')
        orig = np.copy(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, channel]
        img2 = cv2.imread('orig.png')
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        img2[:, :, 0] = img[:, :]
        img2[:, :, 1] = img[:, :]
        cv2.imwrite('orig_lab.png', img2)
        object_under_test = OrientedGradientCalculator(img, 10, 0)
        img = object_under_test.calculate()
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('post_calculate.png', img)

    def test_calculate_oriented_gradient_vis2(self):
        img = cv2.imread('black_white_halves.jpg')
        orig = np.copy(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, 0]
        object_under_test = OrientedGradientCalculator(img, 12, 90)
        img = object_under_test.calculate()
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('test_img_calc.png', img)
