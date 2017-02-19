import cv2

from segmentation.channel_handlers.abstract.OrientedGradientChannelHandler import OrientedGradientChannelHandler


# pass in BGR img
class OrientedGradientaHandler(OrientedGradientChannelHandler):
    def __init__(self, img, radius, angle):
        OrientedGradientChannelHandler.__init__(self, img, radius, angle)

    def _set_image(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)[:, :, 1]
