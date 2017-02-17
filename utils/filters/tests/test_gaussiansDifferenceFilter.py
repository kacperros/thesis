from copy import copy
from unittest import TestCase
import cv2

from utils.filters.GaussiansDifferenceFilter import GaussiansDifferenceFilter


class TestGaussiansDifferenceFilter(TestCase):
    def setUp(self):
        self.img = cv2.imread('orig.png')
        self.orig = copy(self.img)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        self.img = self.img[:, :, 0]

    def test_test1(self):
        object_under_test = GaussiansDifferenceFilter(self.img, [2, 1], [(0, 0), (0, 0)], (0, 0))
        filtered = object_under_test.filter()
        cv2.imwrite('DoG_2_1.png', filtered)
