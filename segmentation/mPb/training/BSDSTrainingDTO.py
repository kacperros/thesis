from segmentation.mPb.AngledmPbCalculator import AngledmPBCalculator
import numpy as np


class BSDSTrainingDTO:
    def __init__(self, img, truth, angle):
        self.img = img
        self.truth = truth
        self.angle = angle
        self.calculator = AngledmPBCalculator(np.copy(img), self.angle)

    def prepare(self):
        self.calculator.prepare_gradients()
