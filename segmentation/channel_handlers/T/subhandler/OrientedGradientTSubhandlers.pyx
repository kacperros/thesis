from utils.filters.directed_gaussian.DirectedGaussianFilter import DirectedGaussianFilter
from utils.filters.gaussians_difference.GaussiansDifferenceFilter import GaussiansDifferenceFilter


class OrientedGradientTSubhandler:
    pass


class OrientedGradientTOrientedGaussianSubhandler(OrientedGradientTSubhandler):
    def __init__(self, img, sigma, order, angle):
        self.img = img
        self.sigma = sigma
        self.order = order
        self.angle = angle

    def handle(self):
        filtered = DirectedGaussianFilter(self.img, self.angle, self.sigma, self.order)
        self.img = filtered.filter()
        return [self.img, self.sigma, self.order, self.angle, True]


class OrientedGradientTGaussianDifferenceSubhandler(OrientedGradientTSubhandler):
    def __init__(self, img, sigmas, orders):
        self.img = img
        self.sigmas = sigmas
        self.orders = orders
        self.angles = [0, 0]

    def handle(self):
        filtered = GaussiansDifferenceFilter(self.img, self.sigmas, self.orders, self.angles)
        self.img = filtered.filter()
        return [self.img, self.sigmas, self.orders, self.angles, False]
