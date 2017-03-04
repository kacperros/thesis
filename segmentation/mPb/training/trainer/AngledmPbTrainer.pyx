import os
from time import time

import numpy as np

import constants
from segmentation.mPb.training.bsds.BSDS import BSDS
from segmentation.mPb.training.bsds_dto.BSDSTrainingDTO import BSDSTrainingDTO


class AngledmPbTrainer:
    def __init__(self, angle):
        self.alfas = np.random.rand(4, 3)
        self.bsds = BSDS()
        self.angle = angle

    def prep(self):
        self.prepare_train()
        self.prepare_val()
        self.prepare_test()

    def train(self):
        pass

    def prepare_train(self):
        truths_path = constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_TRUTHS + '/'
        images_path = constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_IMGS + '/'
        processeds_path = constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_PREPROCESSED + '/' + str(self.angle) + '/'
        self.__prepare_from_paths(truths_path, images_path, processeds_path)

    def prepare_val(self):
        truths_path = constants.BSDS_PATH + constants.BSDS_PATH_VAL_TRUTHS + '/'
        images_path = constants.BSDS_PATH + constants.BSDS_PATH_VAL_IMGS + '/'
        processeds_path = constants.BSDS_PATH + constants.BSDS_PATH_VAL_PREPROCESSED + '/' + str(self.angle) + '/'
        self.__prepare_from_paths(truths_path, images_path, processeds_path)

    def prepare_test(self):
        truths_path = constants.BSDS_PATH + constants.BSDS_PATH_TEST_TRUTHS + '/'
        images_path = constants.BSDS_PATH + constants.BSDS_PATH_TEST_IMGS + '/'
        processeds_path = constants.BSDS_PATH + constants.BSDS_PATH_TEST_PREPROCESSED + '/' + str(self.angle) + '/'
        self.__prepare_from_paths(truths_path, images_path, processeds_path)

    def __prepare_from_paths(self, truths_path, images_path, processeds_path):
        dtos = []
        truths_names = os.listdir(truths_path)
        images_names = os.listdir(images_path)
        processeds_paths = [processeds_path + i[0:len(i) - 4] + '/' for i in truths_names]
        for i in range(0, len(truths_names)):
            time_s = time()
            bsds_data = self.bsds.create_from_file(truths_path + truths_names[i], images_path + images_names[i])
            dto = BSDSTrainingDTO(bsds_data['img'], bsds_data['truth'], self.angle, processeds_paths[i])
            dtos.append(dto)
            dto.prepare(False)
            print('Completed in time: ' + str(time() - time_s))

angles = [int(np.rad2deg(np.pi * i / 8)) for i in range(0, 8)]
for angle in angles:
    trainer = AngledmPbTrainer(0)
    trainer.prep()