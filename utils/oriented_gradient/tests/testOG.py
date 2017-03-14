from unittest import TestCase

import cv2
import numpy as np
import time
from scipy import ndimage

from utils.oriented_gradient.OrientedGradientCalculator import OrientedGradientCalculator


class TestOG(TestCase):
    def test_calculate_oriented_gradient_vis(self):
        time_start = time.time()
        channel = 0
        img = cv2.imread('orig.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, channel]
        object_under_test = OrientedGradientCalculator(img,10, 0)
        img = object_under_test.calculate()
        # img = (img/np.amax(img)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('small2.png', img)
        print(time.time() - time_start)

    def test_calculate_oriented_gradient_vis2(self):
        time_start = time.time()
        img = cv2.imread('orig.png')
        orig = np.copy(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = img[:, :, 0]
        object_under_test = OrientedGradientCalculator(img, 10, 45)
        img = object_under_test.calculate()
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('test_img_calc.png', img)
        print(time.time() - time_start)
