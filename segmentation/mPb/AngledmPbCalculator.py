from copy import copy
from multiprocessing import Queue, Process
import cv2
import numpy as np

from segmentation.channel_handlers.OrientedGradientLHandler import OrientedGradientLHandler
from segmentation.channel_handlers.OrientedGradientTHandler import OrientedGradientTHandler
from segmentation.channel_handlers.OrientedGradientaHandler import OrientedGradientaHandler
from segmentation.channel_handlers.OrientedGradientbHandler import OrientedGradientbHandler


def _calculate_with_handler(handler, img, radius, angle, i, j, queue):
    print('Performing for: ' + str(i) + ' ' + str(j))
    queue.put([handler(img, radius, angle).handle(), i, j])
    print('Performed for: ' + str(i) + ' ' + str(j))


# takes normal RGB image
class AngledmPBCalculator:
    def __init__(self, img, angle):
        self.img = img
        self.angle = angle
        self.gradient_result_queue = Queue()
        self.alfas = np.zeros((4, 3))
        self.variables = np.array([[3, 4, 10],  # L
                                   [5, 10, 20],  # a
                                   [5, 10, 20],  # b
                                   [5, 10, 20]])  # T
        self.handlers = [
            OrientedGradientLHandler,
            OrientedGradientaHandler,
            OrientedGradientbHandler,
            OrientedGradientTHandler
        ]
        self.gradients = None
        self.prepared = False

    def calculate(self):
        if not self.prepared:
            raise RuntimeError('Sir, you need to prepare before calculation')
        return np.sum(np.array([self.gradients[i, j] * self.alfas[i, j] for i in range(0, self.alfas.shape[0]) for j in
                                range(0, self.alfas.shape[1])]), axis=0)

    def prepare_gradients(self):
        jobs = []
        gradients = np.zeros((self.alfas.shape[0], self.alfas.shape[1], self.img.shape[0], self.img.shape[1]))
        for i in range(0, self.variables.shape[0]):
            for j in range(0, self.variables.shape[1]):
                p = Process(target=_calculate_with_handler,
                            args=(self.handlers[i], copy(self.img), self.variables[i, j],
                                  self.angle, i, j, self.gradient_result_queue))
                jobs.append(p)
                p.start()
        for i in range(0, 12):
            gradient = self.gradient_result_queue.get()
            gradients[gradient[1], gradient[2]] = gradient[0]
        for job in jobs:
            job.join()
        self.gradients = gradients
        self.prepared = True
        return self.gradients