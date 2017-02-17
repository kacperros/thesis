import utils.basic_operations.rotation as rotation
import numpy as np
from scipy import ndimage as ndim


# expects single channel image
def calculate_oriented_gradient(img, angle, radius):
    orig_rows, orig_cols = img.shape
    img = ndim.rotate(img, angle)
    rows, cols = img.shape
    result = np.zeros((rows, cols))
    radius -= 1
    for i in range(radius, rows - radius + 1):
        for j in range(radius, cols - radius + 1):
            result[i, j] = calculate_single_oriented_gradient(img[i - radius:i + radius, j - radius:j + radius], radius)
    result = rotation.center_rotate(result, -1 * angle)
    return result


def calculate_single_oriented_gradient(img_part, radius):
    top, dists = np.histogram(img_part[0:radius, :], 16, (0, 255))
    bottom, dists2 = np.histogram(img_part[radius:, :], 16, (0, 255))
    return 0.5 * np.sum(np.nan_to_num(np.divide(np.subtract(top, bottom) ** 2, np.add(top, bottom))))
