from unittest import TestCase

import cv2
import numpy as np
import time

from segmentation.channel_handlers.OrientedGradientTHandler import OrientedGradientTHandler


class TestTHandler(TestCase):
    def setUp(self):
        self.img = cv2.imread('orig.png')

    def test_calculate_oriented_gradient(self):
        for i in range(0, 12):
            object_under_test = OrientedGradientTHandler(np.copy(self.img), 10, 0)
            res = object_under_test.handle()
            cv2.imwrite('result' + str(i) + '.png', res)
