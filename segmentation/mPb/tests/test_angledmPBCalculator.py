from unittest import TestCase
import cv2

from segmentation.mPb.AngledmPbCalculator import AngledmPBCalculator


class TestAngledmPBCalculator(TestCase):
    def test_calculate(self):
        img = cv2.imread('orig.png')
        calculator = AngledmPBCalculator(img, 0)
        res = calculator.calculate()
        print('Done')