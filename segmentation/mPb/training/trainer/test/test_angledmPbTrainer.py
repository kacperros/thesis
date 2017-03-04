from time import time
from unittest import TestCase

import cv2

import constants
from segmentation.mPb.training.bsds.BSDS import BSDS
from segmentation.mPb.training.bsds_dto.BSDSTrainingDTO import BSDSTrainingDTO
from segmentation.mPb.training.trainer.AngledmPbTrainer import AngledmPbTrainer


class TestAngledmPbTrainer(TestCase):
    def test_prepare(self):
        test_object = AngledmPbTrainer(90)
        test_object.prepare_train()
        print('Done')

    def test_prep_singleimg(self):
        time_s = time()
        truth = BSDS().truth_from_file(constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_TRUTHS + '/' + '23080.mat')
        img = cv2.imread(constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_IMGS + '/' + '23080.jpg')
        dto = BSDSTrainingDTO(img, truth, 90, constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_IMGS + '/' + '23080/')
        dto.prepare(False)
        print(time() - time_s)
