from time import time
from unittest import TestCase

import cv2
import numpy as np

from segmentation.channel_handlers.Lab.L.OrientedGradientLHandler import OrientedGradientLHandler
from segmentation.channel_handlers.T.OrientedGradientTHandler import OrientedGradientTHandler


class TestTHandler(TestCase):
    def setUp(self):
        self.img = cv2.imread('orig.png')

    def test_calculate_oriented_gradient(self):
        time_s = time()
        object_under_test = OrientedGradientTHandler(np.copy(self.img), 10, 45)
        res = object_under_test.handle()
        img = res.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite('resultT.png', img)
        print(time() - time_s)

    def test_calculate_oriented_gradientL(self):
        time_s = time()
        object_under_test = OrientedGradientLHandler(np.copy(self.img), 12, 45)
        res = object_under_test.handle()
        cv2.imwrite('resultL.png', res)
        print(time() - time_s)
