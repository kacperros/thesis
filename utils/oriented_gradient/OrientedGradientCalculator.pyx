import multiprocessing
import cv2
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
        self.bins = 96

    def _rotate(self, angle):
        self._rotated = np.copy(self.original)
        self._extend_img(self.original.shape[0], self.original.shape[1])
        self._rotated = ndim.rotate(self._rotated, angle)

    def _operate(self):
        integral_images = self._integrate_images()
        binned_images = self._calculate_bins(integral_images)
        binned_images = np.nan_to_num(binned_images)
        output = 0.5 * np.sum(binned_images, axis=0)
        output = sig_filters.savgol_filter(output, 5, 2, axis=0)
        output = sig_filters.savgol_filter(output, 5, 2, axis=1)
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
            mask = (intervals[i] <= images[i]) & (images[i] <= intervals[i + 1])
            images[i][np.logical_not(mask)] = 0
            images[i][mask] = 1
            images[i] = integral_img.integral_image(images[i])
        return images

    def _calculate_bins(self, integral_images):
        k = 0
        r = self.radius
        copies = np.copy(integral_images)
        for image in integral_images:
            for i in range(r, image.shape[0] - r):
                for j in range(r, image.shape[1] - r):
                    top_bin = image[i - r, j - r] + image[i - 1, j + r - 1] \
                              - image[i - 1, j - r] - image[i - r, j + r - 1]
                    bottom_bin = image[i, j - r] + image[i + r - 1, j + r - 1] \
                                 - image[i + r - 1, j - r] - image[i, j + r - 1]
                    bins = top_bin + bottom_bin if top_bin + bottom_bin != 0 else 1
                    copies[k, i, j] = (top_bin - bottom_bin) ** 2 / bins
            k += 1
        return copies
