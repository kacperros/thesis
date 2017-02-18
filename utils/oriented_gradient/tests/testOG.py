from unittest import TestCase

import cv2
import numpy as np

from utils.oriented_gradient.OrientedGradientCalculator import OrientedGradientCalculator


class TestOG(TestCase):
    def setUp(self):
        self.img = cv2.imread('gaussian_test_img.png')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.img = self.img[:, :, 0]

    def test_calculate_oriented_gradient(self):
        object_under_test = OrientedGradientCalculator(self.img, 6, 0)
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
        img = cv2.imread('orig.png')
        orig = np.copy(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, 0]
        object_under_test = OrientedGradientCalculator(img, 10, 90)
        img = object_under_test.calculate()
        cv2.imwrite('post_calculate2.png', img)
