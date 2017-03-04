import numpy as np
import os
from segmentation.mPb.calculator.AngledmPbCalculator import AngledmPBCalculator


class BSDSTrainingDTO:
    def __init__(self, img, truth, angle, folder_path):
        self.img = img
        self.truth = truth
        self.angle = angle
        self.calculator = AngledmPBCalculator(np.copy(img), self.angle)
        self.folder_path = folder_path
        self.gradients = None

    def prepare(self, from_file):
        if not from_file:
            print('Started gradient calculation for ' + self.folder_path)
            self.gradients = self.calculator.prepare_gradients()
            self.save_to_file()
            print('Gradients saved')
        else:
            self.read_from_file()

    def save_to_file(self):
        os.makedirs(self.folder_path)
        np.save(self.folder_path + '/truth.npy', self.truth)
        np.save(self.folder_path + '/gradients.npy', self.gradients)

    def read_from_file(self):
        self.gradients = np.load(self.folder_path + '/gradients.npy')
        self.truth = np.load(self.folder_path + '/truth.npy')
