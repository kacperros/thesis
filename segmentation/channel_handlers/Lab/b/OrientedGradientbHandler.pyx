import cv2

from segmentation.channel_handlers.channel.OrientedGradientChannelHandler import OrientedGradientChannelHandler


class OrientedGradientbHandler(OrientedGradientChannelHandler):
    def __init__(self, img, radius, angle):
        OrientedGradientChannelHandler.__init__(self, img, radius, angle)

    def _set_image(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)[:, :, 2]
