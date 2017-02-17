import numpy as np
import scipy.ndimage.filters as filters

import utils.basic_operations.rotation as rotation


# gft.oriented_gaussian(gimage, 1, (0, 1), 60)
def oriented_gaussian(image, sigma, order, angle_anticlockwise, mode='nearest', cval=0.0, truncate=4.0):
    rows, cols = image.shape
    rotated = rotation.center_rotate(image, angle_anticlockwise)
    output_rotated = np.zeros(rotated.shape)
    filters.gaussian_filter(rotated, sigma, order, output_rotated, mode, cval, truncate)
    output_uncropped = rotation.center_rotate(output_rotated, -1 * angle_anticlockwise)
    rows2, cols2 = output_uncropped.shape
    output = output_uncropped[int(np.ceil(rows2 / 2 - rows / 2)): int(np.ceil(rows2 / 2 + rows / 2)),
             int(np.ceil(cols2 / 2 - cols / 2)): int(np.ceil(cols2 / 2 + cols / 2))]
    minimum = np.amin(output)
    output = output + np.abs(minimum)
    output = np.around(output)
    return output