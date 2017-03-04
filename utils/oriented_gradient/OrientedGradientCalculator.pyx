import multiprocessing

import numpy as np
import scipy.signal as sig_filters
from scipy import ndimage as ndim
import skimage.transform.integral as integral_img
from utils.RotationAdapter.RotationAdapter import RotationAdapter


class OrientedGradientCalculator(RotationAdapter):
    def __init__(self, img, radius, angle):
        RotationAdapter.__init__(self, img)
        self.radius = radius - 1
        self.angle = angle
        self.bins = 128

    def _rotate(self, angle):
        self._rotated = np.copy(self.original)
        self._extend_img(self.original.shape[0], self.original.shape[1])
        self._rotated = ndim.rotate(self._rotated, angle)

    def _operate(self):
        output = np.zeros((self._rotated.shape[0], self._rotated.shape[1]))
        integral_images = self._integrate_images()
        binned_images = self._calculate_bins(integral_images)
        output = 0.5 * np.sum(binned_images)
        # output = sig_filters.savgol_filter(output, 3, 2, axis=0)
        # output = sig_filters.savgol_filter(output, 3, 2, axis=1)
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
        intervals = np.array([i * (255 / self.bins) for i in range(0, self.bins + 1)])
        images = np.array([np.copy(self._rotated).astype(np.float) for i in range(0, self.bins)])
        for i in range(0, self.bins):
            images[i][(intervals[i] > images[i]) | (intervals[i + 1] < images[i])] = 0
            images[i][(intervals[i] < images[i]) & (images[i] < intervals[i + 1])] = 1
            images[i] = integral_img.integral_image(images[i])
        return images

    def _calculate_bins(self, integral_images):
        for image in integral_images:
            for i in range(self.radius, image.shape[0] - self.radius):
                for j in range(self.radius, image.shape[1] - self.radius):
                    image[i, j] = integral_img.integrate(image, (i - self.radius, j - self.radius),
                                                         (i + self.radius - 1, j + 1)) \
                                  - integral_img.integrate(image, (i - self.radius, j),
                                                           (i + self.radius - 1, j + self.radius - 1))

        return integral_images
