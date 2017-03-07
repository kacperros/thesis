from multiprocessing.pool import Pool

import numpy as np
import scipy.signal as sig_filters
from scipy import ndimage as ndim

from utils.RotationAdapter.RotationAdapter import RotationAdapter


class OrientedGradientCalculator(RotationAdapter):
    def __init__(self, img, radius, angle):
        RotationAdapter.__init__(self, img)
        self.radius = radius - 1
        self.angle = angle
        self.hist_bins = 256

    def _rotate(self, angle):
        self._rotated = np.copy(self.original)
        self._extend_img(self.original.shape[0], self.original.shape[1])
        self.mask = np.full(self._rotated.shape, True)
        self._rotated = ndim.rotate(self._rotated, angle)
        self.mask = ndim.rotate(self.mask, angle)

    def _operate(self):
        out2 = np.array([(j, i) for j in range(0, self._rotated.shape[0]) for i in range(0, self._rotated.shape[1])])
        with Pool(8) as p:
            result = p.map(self._calculate_single_oriented_gradient2, out2)
            output = np.reshape(result, self._rotated.shape)
            output = np.around(output, 3)
            output = sig_filters.savgol_filter(output, 3, 2, axis=0)
            output = sig_filters.savgol_filter(output, 3, 2, axis=1)
            self._rotated = output

    def calculate(self):
        return self.adapt(self.angle)

    def _calculate_single_oriented_gradient2(self, args):
        row, col = args
        if not self.mask[row, col]:
            return 0
        top, dists = np.histogram(self._rotated[row - self.radius:row, col - self.radius:col + self.radius], 256,
                                  (0, 255))
        bottom, dists2 = np.histogram(self._rotated[row:row + self.radius, col - self.radius:col + self.radius], 256,
                                      (0, 255))
        divisor = np.add(top, bottom)
        divisor[divisor == 0] = 1
        return 0.5 * np.sum(np.divide(np.subtract(top, bottom) ** 2, divisor))

    def _extend_img(self, orig_rows, orig_cols):
        self._rotated = np.vstack((self._rotated[range(self.radius - 1, -1, -1), :],
                                   self._rotated,
                                   self._rotated[range(orig_rows - 1, orig_rows - self.radius - 1, -1), :]))

        self._rotated = np.hstack((np.reshape(self._rotated[:, range(self.radius - 1, -1, -1)],
                                              (self._rotated.shape[0], self.radius)),
                                   self._rotated,
                                   np.reshape(self._rotated[:, range(orig_cols - 1, orig_cols - self.radius - 1, -1)],
                                              (self._rotated.shape[0], self.radius))))
