from unittest import TestCase

import cv2
import numpy as np
import time

from segmentation.channel_handlers.OrientedGradientLHandler import OrientedGradientLHandler
from segmentation.channel_handlers.OrientedGradientTHandler import OrientedGradientTHandler


class TestTHandler(TestCase):
    def setUp(self):
        self.img = cv2.imread('orig.png')

    def test_calculate_oriented_gradient(self):
        object_under_test = OrientedGradientTHandler(np.copy(self.img), 10, 0)
        res = object_under_test.handle()
        img = res.astype(np.uint8)
        img = img * 8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('resultT.png', img)

    def test_calculate_oriented_gradientL(self):
        object_under_test = OrientedGradientLHandler(np.copy(self.img), 10, 0)
        res = object_under_test.handle()
        cv2.imwrite('resultL.png', res)
