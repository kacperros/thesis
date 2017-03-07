import time
from copy import copy
from unittest import TestCase

import cv2

from utils.filters.DirectedGaussian.DirectedGaussianFilter import DirectedGaussianFilter


class TestDirectedGaussianFilter(TestCase):
    def setUp(self):
        self.img = cv2.imread('orig.png')
        self.orig = copy(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.img = self.img[:, :, 0]

    def test_test1(self):
        object_under_test = DirectedGaussianFilter(self.img, 90, 1, (0, 1))
        start_t = time.time()
        filtered = object_under_test.filter()
        print(time.time() - start_t)
        cv2.imwrite('90_gaussian_1_01.png', filtered)

    def test_test2(self):
        object_under_test = DirectedGaussianFilter(self.img, 0, 1, (0, 1))
        filtered = object_under_test.filter()
        cv2.imwrite('0_gaussian_1_01.png', filtered)
