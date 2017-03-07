import multiprocessing
from time import time

import cv2
import numpy as np
from sklearn.cluster import KMeans

from segmentation.channel_handlers.Channel.OrientedGradientChannelHandler import OrientedGradientChannelHandler
from segmentation.channel_handlers.T.subhandler.OrientedGradientTSubhandlers import \
    OrientedGradientTOrientedGaussianSubhandler, \
    OrientedGradientTGaussianDifferenceSubhandler
from utils.oriented_gradient.OrientedGradientCalculator import OrientedGradientCalculator


class OrientedGradientTHandler(OrientedGradientChannelHandler):
    def __init__(self, img, radius, angle):
        OrientedGradientChannelHandler.__init__(self, img, radius, angle)
        self._subhandler_result = multiprocessing.Queue()
        self.images_with_attributes = []
        self.responses = np.full((self.img.shape[0], self.img.shape[1], 17), 0)

    def _set_image(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)[:, :, 0]

    def handle(self):
        start_t = time()
        self.__parallel_filter()
        print('Parallel filtering took ', time() - start_t)
        start_t = time()
        self.__combine_images()
        print('Combining took ', time() - start_t)
        start_t = time()
        texton_img = self.__set_textons()
        print('kMeans took ', time() - start_t)
        start_t = time()
        res = OrientedGradientCalculator(texton_img, self.radius, self.angle).calculate()
        print('Oriented gradient took ', time() - start_t)
        return res

    def __parallel_filter(self):
        orders = [(1, 1) for i in range(0, 8)]
        orders.extend([(2, 2) for i in range(0, 8)])
        angles = [int(np.rad2deg(np.pi * (i / 8))) for i in range(0, 8)]
        angles.extend(angles)
        jobs = []
        for i in range(0, 16):
            p = multiprocessing.Process(
                target=OrientedGradientTOrientedGaussianSubhandler(np.copy(self.img), 1, orders[i], angles[i]).handle,
                args=(self._subhandler_result,))
            jobs.append(p)
            p.start()
        p = multiprocessing.Process(
            target=OrientedGradientTGaussianDifferenceSubhandler(np.copy(self.img), (2, 1), [(0, 0), (0, 0)]).handle,
            args=(self._subhandler_result,))
        p.start()
        jobs.append(p)
        for i in range(0, 17):
            self.images_with_attributes.append(self._subhandler_result.get())
        for job in jobs:
            job.join()

    def __combine_images(self):
        rows, cols = self.img.shape[0], self.img.shape[1]
        self.responses = np.array([[[self.images_with_attributes[k][0][i, j] for k in range(0, 17)]
                           for j in range(0, cols)] for i in range(0, rows)])

    def __set_textons(self):
        responses = np.reshape(self.responses,
                               (self.responses.shape[0] * self.responses.shape[1], self.responses.shape[2]))
        kMeans = KMeans(n_clusters=32, n_jobs=-1).fit_predict(responses)
        return np.reshape(kMeans, (self.responses.shape[0], self.responses.shape[1]))
