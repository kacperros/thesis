import scipy.ndimage.filters as filters
import numpy as np
from utils.filters.DirectedFilter import DirectedFilter


class DirectedGaussianFilter(DirectedFilter):
    def __init__(self, image, angle, sigma, order):
        super().__init__(image, angle)
        self.sigma = sigma
        self.order = order

    def _directed_filter(self):
        self._rotated = filters.gaussian_filter(self._rotated, self.sigma, self.order, output=np.float64, mode='nearest', cval=0.0, truncate=4.0)

    def _post_process(self):
        minimum = np.amin(self.img)
        self.img = self.img + np.abs(minimum)
        self.img = np.around(self.img)
        return self.img
