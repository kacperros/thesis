from unittest import TestCase

import cv2
import time

from segmentation.mPb.calculator.AngledmPbCalculator import AngledmPBCalculator


class TestAngledmPBCalculator(TestCase):
    def test_calculate(self):
        time_s = time.time()
        img = cv2.imread('orig.png')
        calculator = AngledmPBCalculator(img, 60)
        res = calculator.prepare_gradients()
        print(time.time() - time_s)
