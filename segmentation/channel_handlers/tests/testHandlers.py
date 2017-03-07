from time import time
from unittest import TestCase

import cv2
import numpy as np

from segmentation.channel_handlers.Lab.L.OrientedGradientLHandler import OrientedGradientLHandler
from segmentation.channel_handlers.T.handler.OrientedGradientTHandler import OrientedGradientTHandler


class TestTHandler(TestCase):
    def setUp(self):
        self.img = cv2.imread('orig.png')

    def test_calculate_oriented_gradient(self):
        object_under_test = OrientedGradientTHandler(np.copy(self.img), 10, 0)
        start = time()
        res = object_under_test.handle()
        print(time() - start)
        img = res.astype(np.uint8)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('resultT.png', img)

    def test_calculate_oriented_gradientL(self):
        object_under_test = OrientedGradientLHandler(np.copy(self.img), 10, 45)
        start = time()
        res = object_under_test.handle()
        print(time() - start)
        cv2.imwrite('resultL.png', res)
