import multiprocessing

import numpy as np

from segmentation.mPb.training.BSDS import BSDS
from segmentation.mPb.training.BSDSTrainingDTO import BSDSTrainingDTO


class AngledmPbTrainer:
    def __init__(self, angle):
        self.alfas = np.random.rand(4, 3)
        self.bsds = BSDS()
        self.angle = angle
        self.gradients = None

    def prepare(self):
        training_set = self.bsds.training_set
        dtos = [BSDSTrainingDTO(dictn['img'], dictn['truth'], self.angle) for file_name, dictn in training_set.items()]
        jobs = []
        for dto in dtos[1:3]:
            p = multiprocessing.Process(target=dto.prepare)
            p.start()
            jobs.append(p)
        for job in jobs:
            job.join()