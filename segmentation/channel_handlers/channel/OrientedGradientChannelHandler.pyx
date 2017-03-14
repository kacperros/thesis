from segmentation.channel_handlers.abstract.ChannelHandler import ChannelHandler


# Pass RGB image
from utils.oriented_gradient.OrientedGradientCalculator import OrientedGradientCalculator


class OrientedGradientChannelHandler(ChannelHandler):
    def __init__(self, img, radius, angle):
        ChannelHandler.__init__(self, img)
        self._set_image()
        self.radius = radius
        self.angle = angle

    def _set_image(self):
        pass

    def handle(self):
        gradinet_calculator = OrientedGradientCalculator(self.img, self.radius, self.angle, 32)
        return gradinet_calculator.calculate()
