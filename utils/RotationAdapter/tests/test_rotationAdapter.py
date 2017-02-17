from unittest import TestCase
import cv2
import numpy
from utils.RotationAdapter.RotationAdapter import RotationAdapter


class TestRotationAdapter(TestCase):
    def setUp(self):
        self.img = cv2.imread('gaussian_test_img.png')
        self.adapter = RotationAdapter(self.img)

    def test_adapt1(self):
        new_img = self.adapter.adapt(90)
        numpy.testing.assert_array_equal(self.img, new_img)
