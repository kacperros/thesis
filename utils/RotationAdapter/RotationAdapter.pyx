import numpy as np
from scipy import ndimage as ndim
import cv2

class RotationAdapter:
    def __init__(self, image):
        self.original = image
        self._rotated = np.copy(self.original)

    def _rotate(self, angle):
        self._rotated = ndim.rotate(np.copy(self.original), angle)

    def _operate(self):
        pass

    def _rotate_back(self, angle):
        self._rotated = ndim.rotate(self._rotated, -1 * angle)
        orows, ocols = self.original.shape[0], self.original.shape[1]
        nrows, ncols = self._rotated.shape[0], self._rotated.shape[1]
        return self._rotated[int(np.ceil(nrows / 2 - orows / 2)): int(np.ceil(nrows / 2 + orows / 2)),
               int(np.ceil(ncols / 2 - ocols / 2)): int(np.ceil(ncols / 2 + ocols / 2))]

    def adapt(self, angle):
        self._rotate(angle)
        self._operate()
        return self._rotate_back(angle)
