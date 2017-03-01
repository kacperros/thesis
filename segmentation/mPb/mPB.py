import multiprocessing

import numpy as np

import constants
from segmentation.mPb.AngledmPbTrainer import AngledmPbTrainer


class MPbCalculator:
    def __init__(self, img):
        self.img = img
        self.angles = [int(np.rad2deg(np.pi * (i / 8))) for i in range(0, 8)]
        self.angled_results = multiprocessing.Queue()

    def calculate(self):
        pass


