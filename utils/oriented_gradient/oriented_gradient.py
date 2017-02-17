import utils.basic_operations.rotation as rotation
import numpy as np


# expects single channel image
def calculate_oriented_gradient(img, angle, radius):
    img = rotation.center_rotate(img, angle)
    rows, cols, channels = img.shape
    result = np.zeros((rows, channels))
    for i in range(radius, rows-radius):
        for j in range(radius, cols-radius):
            result[i,j] = calculate_single_oriented_gradient(img[i-radius:i+radius+1, j-radius:j+radius+1], radius)
    result = rotation.center_rotate(result, -1 * angle)
    return result


def calculate_single_oriented_gradient(img_part, radius):
    hist_top = np.histogram()
    return []