import time

import numpy as np
import scipy.signal as sig_filters
from scipy import ndimage as ndim

from utils.RotationAdapter.RotationAdapter import RotationAdapter


class OrientedGradientCalculator(RotationAdapter):
    def __init__(self, img, radius, angle):
        RotationAdapter.__init__(self, img)
        self.radius = radius
        self.angle = angle
        self.bins = 16

    def _rotate(self, angle):
        self._rotated = np.copy(self.original)
        start_time = time.time()
        self._extend_img(self.original.shape[0], self.original.shape[1])
        self._rotated = (self._rotated / 256) * self.bins
        self.mask = np.full(self._rotated.shape, True)
        start_time = time.time()
        self._rotated = ndim.rotate(self._rotated, angle)
        self.mask = ndim.rotate(self.mask, angle)

    def _operate(self):
        start_time = time.time()
        self._rotated[self._rotated >= self.bins] = self.bins -1
        integral_images = self._integrate_images()
        start_time = time.time()
        output = self._calculate_bins(integral_images)
        start_time = time.time()
        output = sig_filters.savgol_filter(output, 3, 2, axis=0)
        output = sig_filters.savgol_filter(output, 3, 2, axis=1)
        self._rotated = output

    def calculate(self):
        return self.adapt(self.angle)

    def _extend_img(self, orig_rows, orig_cols):
        self._rotated = np.vstack((self._rotated[range(self.radius - 1, -1, -1), :],
                                   self._rotated,
                                   self._rotated[range(orig_rows - 1, orig_rows - self.radius - 1, -1), :]))

        self._rotated = np.hstack((np.reshape(self._rotated[:, range(self.radius - 1, -1, -1)],
                                              (self._rotated.shape[0], self.radius)),
                                   self._rotated,
                                   np.reshape(self._rotated[:, range(orig_cols - 1, orig_cols - self.radius - 1, -1)],
                                              (self._rotated.shape[0], self.radius))))

    def _integrate_images(self):
        binned = np.zeros((self._rotated.shape[0], self._rotated.shape[1], self.bins))
        for i in range(0, self._rotated.shape[0]):
            for j in range(0, self._rotated.shape[1]):
                binned[i, j, int(self._rotated[i, j])] = 1
        for i in range(0, 2):
            binned = np.cumsum(binned, i)
        return binned

    def _calculate_bins(self, img):
        r = self.radius
        res = np.copy(self._rotated)
        for i in range(r, img.shape[0] - r):
            for j in range(r, img.shape[1] - r):
                res[i, j] = self._calc_gradient_at(img, i, j, r)
        return res

    def _calc_gradient_at(self, image, i, j, r):
        if not self.mask[i,j]:
            return 0
        top = image[i - r, j - r] + image[i - 1, j + r - 1] \
              - image[i - 1, j - r] - image[i - r, j + r - 1]
        bottom = image[i, j - r] + image[i + r - 1, j + r - 1] \
                 - image[i + r - 1, j - r] - image[i, j + r - 1]
        divisor = top + bottom
        divisor[divisor == 0] = 1
        return 0.5 * np.sum((top - bottom) ** 2 / divisor)

