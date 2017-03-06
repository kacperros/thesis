import multiprocessing
import cv2
import numpy as np
import scipy.signal as sig_filters
import time
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
        start_time = time.time()
        self._extend_img(self.original.shape[0], self.original.shape[1])
        print('Extending took: ', time.time() - start_time)
        start_time = time.time()
        self._rotated = ndim.rotate(self._rotated, angle)
        print('Rotation took: ', time.time() - start_time)

    def _operate(self):
        start_time = time.time()
        integral_images = self._integrate_images()
        print('Integrals took:', time.time() - start_time)
        start_time = time.time()
        binned_images = self._calculate_bins(integral_images)
        print('Binning took:', time.time() - start_time)
        start_time = time.time()
        output = 0.5 * np.sum(binned_images, axis=0)
        print('Summing took:', time.time() - start_time)
        start_time = time.time()
        output = sig_filters.savgol_filter(output, 5, 2, axis=0)
        output = sig_filters.savgol_filter(output, 5, 2, axis=1)
        print('Sav-Gol took:', time.time() - start_time)
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
        images = np.array(
            [self.__mask_img(np.copy(self._rotated).astype(np.float), intervals[i], intervals[i + 1]) for i in
             range(0, self.bins)])
        return images

    def __mask_img(self, img, bot, top):
        mask = (bot <= img) & (img <= top)
        img[np.logical_not(mask)] = 0
        img[mask] = 1
        return integral_img.integral_image(img)

    def _calculate_bins(self, integral_images):
        with multiprocessing.Pool(processes=4) as pool:
            return pool.map(self._calculate_bin, integral_images)

    def _calculate_bin(self, image):
        return [
            [self._calc_gradient_at(image, i, j, self.radius) for j in range(self.radius, image.shape[1] - self.radius)]
            for i in
            range(self.radius, image.shape[0] - self.radius)]

    def _calc_gradient_at(self, image, i, j, r):
        top_bin = image[i - r, j - r] + image[i - 1, j + r - 1] \
                  - image[i - 1, j - r] - image[i - r, j + r - 1]
        bottom_bin = image[i, j - r] + image[i + r - 1, j + r - 1] \
                     - image[i + r - 1, j - r] - image[i, j + r - 1]
        return (top_bin - bottom_bin) ** 2 / (
            top_bin + bottom_bin) if top_bin != 0 or bottom_bin != 0 else 0
