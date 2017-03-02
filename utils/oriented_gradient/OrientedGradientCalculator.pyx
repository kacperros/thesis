import multiprocessing

import numpy as np
import scipy.signal as sig_filters
from scipy import ndimage as ndim

from utils.RotationAdapter.RotationAdapter import RotationAdapter


class OrientedGradientCalculator(RotationAdapter):
    def __init__(self, img, radius, angle):
        RotationAdapter.__init__(self, img)
        self.radius = radius - 1
        self.angle = angle
        self.parts_queue = multiprocessing.Queue()

    def _rotate(self, angle):
        self._rotated = np.copy(self.original)
        self._extend_img(self.original.shape[0], self.original.shape[1])
        self._rotated = ndim.rotate(self._rotated, angle)

    def _operate(self):
        output = np.zeros((self._rotated.shape[0], self._rotated.shape[1]))
        rows, cols = self._rotated.shape[0], self._rotated.shape[1]
        row_bounds = [0, int(rows / 2), rows]
        col_bounds = [0, int(cols / 2), cols]
        jobs = []
        for i in range(0, 2):
            for j in range(0, 2):
                p = multiprocessing.Process(target=self.calculate_image_part, args=(row_bounds[i], row_bounds[i + 1],
                                                                                    col_bounds[j], col_bounds[j + 1]))
                p.start()
                jobs.append(p)
        for k in range(0, 4):
            res = self.parts_queue.get()
            output[res[1]:res[2], res[3]:res[4]] = res[0]
        for job in jobs:
            job.join()
        output = np.around(output, 3)
        output = sig_filters.savgol_filter(output, 3, 2, axis=0)
        output = sig_filters.savgol_filter(output, 3, 2, axis=1)
        self._rotated = output

    def calculate_image_part(self, row_start, row_end, col_start, col_end):
        output = np.zeros((row_end - row_start, col_end - col_start), np.float)
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                output[i - row_start, j - col_start] = self._calculate_single_oriented_gradient(
                    self._rotated[i - self.radius:i + self.radius, j - self.radius:j + self.radius])
        self.parts_queue.put([output, row_start, row_end, col_start, col_end])

    def calculate(self):
        return self.adapt(self.angle)

    def _calculate_single_oriented_gradient(self, img_part):
        top, dists = np.histogram(img_part[0:self.radius, :], 256, (0, 255))
        bottom, dists2 = np.histogram(img_part[self.radius:, :], 256, (0, 255))
        return 0.5 * np.sum(np.nan_to_num(np.divide(np.subtract(top, bottom) ** 2, np.add(top, bottom))))

    def _extend_img(self, orig_rows, orig_cols):
        self._rotated = np.vstack((self._rotated[range(self.radius - 1, -1, -1), :],
                                   self._rotated,
                                   self._rotated[range(orig_rows - 1, orig_rows - self.radius - 1, -1), :]))

        self._rotated = np.hstack((np.reshape(self._rotated[:, range(self.radius - 1, -1, -1)],
                                              (self._rotated.shape[0], self.radius)),
                                   self._rotated,
                                   np.reshape(self._rotated[:, range(orig_cols - 1, orig_cols - self.radius - 1, -1)],
                                              (self._rotated.shape[0], self.radius))))
