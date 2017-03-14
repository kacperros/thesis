from multiprocessing import Pool

import cv2
import numpy as np
import time
from sklearn.cluster import KMeans

from segmentation.channel_handlers.T.subhandler.OrientedGradientTSubhandlers import OrientedGradientTOrientedGaussianSubhandler, \
    OrientedGradientTGaussianDifferenceSubhandler
from segmentation.channel_handlers.channel.OrientedGradientChannelHandler import OrientedGradientChannelHandler
from utils.oriented_gradient.OrientedGradientCalculator import OrientedGradientCalculator


class OrientedGradientTHandler(OrientedGradientChannelHandler):
    def __init__(self, img, radius, angle):
        OrientedGradientChannelHandler.__init__(self, img, radius, angle)
        self.images_with_attributes = []
        self.responses = np.full((self.img.shape[0], self.img.shape[1], 17), 0)

    def _set_image(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)[:, :, 0]

    def handle(self):
        self.__parallel_filter()
        texton_img = self.__set_textons()
        texton_img = (texton_img / 32) * 255
        return OrientedGradientCalculator(texton_img, self.radius, self.angle, 256).calculate()

    def __parallel_filter(self):
        orders = [(1, 1) for i in range(0, 8)]
        orders.extend([(2, 2) for i in range(0, 8)])
        angles = [int(np.rad2deg(np.pi * (i / 8))) for i in range(0, 8)]
        angles.extend(angles)
        handlers = [OrientedGradientTOrientedGaussianSubhandler(np.copy(self.img), 1, orders[i], angles[i]) for i in range(0, 16)]
        handlers.append(OrientedGradientTGaussianDifferenceSubhandler(np.copy(self.img), (2, 1), [(0, 0), (0, 0)]))
        with Pool(4) as pool:
            self.images_with_attributes = pool.map(self._perform_handler, handlers)
            self.__combine_images()

    def _perform_handler(self, handler):
        return handler.handle()

    def __combine_images(self):
        size = (self.images_with_attributes[0][0].shape[0] , self.images_with_attributes[0][0].shape[1], 1)
        self.responses = np.reshape(self.images_with_attributes[0][0], size)
        for k in range(1, 17):
            self.responses = np.append(self.responses, np.reshape(self.images_with_attributes[k][0], size), axis=2)

    def __set_textons(self):
        responses = np.reshape(self.responses,
                               (self.responses.shape[0] * self.responses.shape[1], self.responses.shape[2]))
        kMeans = KMeans(n_clusters=32, n_jobs=-1, max_iter=4).fit_predict(responses)
        return np.reshape(kMeans, (self.responses.shape[0], self.responses.shape[1]))
