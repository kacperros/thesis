from utils.filters.DirectedGaussian.DirectedGaussianFilter import DirectedGaussianFilter
from utils.filters.GaussiansDifference.GaussiansDifferenceFilter import GaussiansDifferenceFilter


class OrientedGradientTSubhandler:
    pass


class OrientedGradientTOrientedGaussianSubhandler(OrientedGradientTSubhandler):
    def __init__(self, img, sigma, order, angle):
        self.img = img
        self.sigma = sigma
        self.order = order
        self.angle = angle

    def handle(self, queue):
        filtered = DirectedGaussianFilter(self.img, self.angle, self.sigma, self.order)
        self.img = filtered.filter()
        # print('Done DG ' + str(self.sigma) + ' ' + str(self.order) + ' ' + str(self.angle))
        queue.put([self.img, self.sigma, self.order, self.angle, True])
        return 0


class OrientedGradientTGaussianDifferenceSubhandler(OrientedGradientTSubhandler):
    def __init__(self, img, sigmas, orders):
        self.img = img
        self.sigmas = sigmas
        self.orders = orders
        self.angles = [0, 0]

    def handle(self, queue):
        filtered = GaussiansDifferenceFilter(self.img, self.sigmas, self.orders, self.angles)
        # print('Done GD' + str(self.sigmas) + ' ' + str(self.orders) + ' ' + str(self.angles))
        self.img = filtered.filter()
        queue.put([self.img, self.sigmas, self.orders, self.angles, False])
        return 0
