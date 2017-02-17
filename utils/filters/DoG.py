import numpy as np

import utils.filters.gaussian as gaussians


# gft.gaussians_difference(gimage, [2, 1], [(0, 0), (0, 0)], [0, 0])
def gaussians_difference(image, sigma, order, angle_aclockwise):
    images = []
    if len(sigma) != len(order) or len(order) != len(angle_aclockwise) or len(sigma) != len(angle_aclockwise):
        raise ValueError("sigma, order, angle_clockwise must have the same number of elements")
    for i in range(0, len(sigma)):
        images.append(gaussians.oriented_gaussian(image, sigma[i], order[i], angle_aclockwise[i]))
    output_image = images[0]
    for i in range(1, len(images)):
        output_image = np.subtract(output_image, images[i])
    minimum = np.amin(output_image)
    output = output_image + np.abs(minimum)
    output = np.around(output)
    return output
