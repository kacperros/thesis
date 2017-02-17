from unittest import TestCase
import cv2
import utils.oriented_gradient.oriented_gradient as og
import numpy as np


class TestOG(TestCase):
    def test_calculate_single_oriented_gradient(self):
        img = cv2.imread('gaussian_test_img.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, 0]
        distance = og.calculate_single_oriented_gradient(img[1:5, 1:5], 2)
        self.assertAlmostEqual(np.around(distance, 2), 2.67, 2)

    def test_calculate_oriented_gradient1(self):
        img = cv2.imread('gaussian_test_img.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, 0]
        distances = og.calculate_oriented_gradient(img, 0, 6)
        self.assertEqual(distances[5, 5], 0)

    def test_calculate_oriented_gradient2(self):
        img = cv2.imread('gaussian_test_img.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, 0]
        distances = og.calculate_oriented_gradient(img, 60, 6)
        self.assertEqual(distances[5, 5], 0)
