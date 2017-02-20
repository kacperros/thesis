import numpy as np
import multiprocessing

import constants


class MPbCalculator:
    def __init__(self, img):
        self.img = img
        self.angles = [int(np.rad2deg(np.pi * (i / 8))) for i in range(0, 8)]
        self.angled_results = multiprocessing.Queue()


    def calculate(self):
        pass


class AngledmPBCalculator:
    def __init__(self, img, angle, result_queue):
        self.img = img
        self.angle = angle
        self.result_queue = result_queue
        self.alfas=[]

    def train(self):
        if not constants.TRAIN_MPB:
            return


    def calculate(self):
        pass