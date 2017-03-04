import os

import cv2
import numpy as np
import scipy.io as scio

import constants


class BSDS:
    def __init__(self):
        self.training_set = None
        self.val_set = None
        self.test_set = None

    def prepare_train(self):
        self.training_set = self.__create_from_files(constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_TRUTHS + '/',
                                                     constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_IMGS + '/')

    def prepare_val(self):
        self.val_set = self.__create_from_files(constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_TRUTHS + '/',
                                                constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_IMGS + '/')

    def prepare_test(self):
        self.test_set = self.__create_from_files(constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_TRUTHS + '/',
                                                 constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_IMGS + '/')

    def __create_from_files(self, truths_dir, imgs_dir):
        truths_names = os.listdir(truths_dir)
        imgs_names = os.listdir(imgs_dir)
        result = {}
        for i in range(0, len(imgs_names)):
            print('Starting truth ' + truths_names[i])
            img = cv2.imread(imgs_dir + imgs_names[i])
            truth = self.truth_from_file(truths_dir + truths_names[i])
            result[imgs_names[i][0:len(imgs_names[i]) - 4]] = {'img': img, 'truth': truth}
            print('Completed truth ' + truths_names[i])
        return result

    def create_from_file(self, truth_path, img_path):
        print('Starting truth at ' + truth_path)
        img = cv2.imread(img_path)
        truth = self.truth_from_file(truth_path)
        print('Done')
        return {'img': img, 'truth': truth}

    def truth_from_file(self, path):
        mat = scio.loadmat(path)
        org = mat['groundTruth'][0][0][0][0][0]
        res = np.full(org.shape, 0, dtype=np.float)
        for k in range(0, len(mat['groundTruth'][0])):
            mat_a = mat['groundTruth'][0][k][0][0][0]
            mat_a = np.array(mat_a, dtype=np.uint8)
            for i in range(1, mat_a.shape[0] - 1):
                for j in range(1, mat_a.shape[1] - 1):
                    if mat_a[i, j] != mat_a[i - 1, j] \
                            or mat_a[i, j] != mat_a[i + 1, j] \
                            or mat_a[i, j] != mat_a[i, j - 1] \
                            or mat_a[i, j] != mat_a[i, j + 1]:
                        res[i, j] += 1
        return res / np.amax(res)
