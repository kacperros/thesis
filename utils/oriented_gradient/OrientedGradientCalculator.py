import numpy as np
import scipy.signal as sig_filters
from utils.RotationAdapter.RotationAdapter import RotationAdapter


class OrientedGradientCalculator(RotationAdapter):
    def __init__(self, img, radius, angle):
        RotationAdapter.__init__(self, img)
        self.radius = radius - 1
        self.angle = angle

    def _operate(self):
        output = np.zeros((self._rotated.shape[0], self._rotated.shape[1]))
        rows, cols = self._rotated.shape[0], self._rotated.shape[1]
        for i in range(self.radius, rows - self.radius + 1):
            for j in range(self.radius, cols - self.radius + 1):
                output[i, j] = self._calculate_single_oriented_gradient(
                    self._rotated[i - self.radius:i + self.radius, j - self.radius:j + self.radius])
        output = np.around(output, 3)
        output = sig_filters.savgol_filter(output, 3, 2, axis=0)
        output = sig_filters.savgol_filter(output, 3, 2, axis=1)
        self._rotated = output

    def calculate(self):
        return self.adapt(self.angle)

    def _calculate_single_oriented_gradient(self, img_part):
        top, dists = np.histogram(img_part[0:self.radius, :], 256, (0, 255))
        bottom, dists2 = np.histogram(img_part[self.radius:, :], 256, (0, 255))
        return 0.5 * np.sum(np.nan_to_num(np.divide(np.subtract(top, bottom) ** 2, np.add(top, bottom))))
