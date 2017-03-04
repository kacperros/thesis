import numpy as np

from utils.filters.Filter import Filter
from utils.filters.directed_gaussian.DirectedGaussianFilter import DirectedGaussianFilter


class GaussiansDifferenceFilter(Filter):
    def __init__(self, img, sigmas, orders, angles):
        Filter.__init__(self, img)
        self.sigmas = sigmas
        self.orders = orders
        self.angles = angles

    def _apply_filter(self):
        images = []
        if not len(self.sigmas) == len(self.orders) == len(self.angles):
            raise ValueError("sigma, order, angle_clockwise must have the same number of elements")
        for i in range(0, len(self.sigmas)):
            images.append(DirectedGaussianFilter(
                np.copy(self.img), self.angles[i], self.sigmas[i], self.orders[i]).filter())
        self.img = images[0]
        for i in range(1, len(images)):
            self.img = np.subtract(self.img, images[i])

    def _post_process(self):
        minimum = np.amin(self.img)
        self.img = np.around(self.img + np.abs(minimum))
        return self.img
