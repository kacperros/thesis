import multiprocessing
import cv2
import constants
import numpy
import os, sys


class AngledmPbCalculatorProvider:
    def __init__(self):
        self.alfas = []
        self.__obtain_alfas()
        self.__truths_q = multiprocessing.Queue()
        self.__truths = []
        self.__processed_images_q = multiprocessing.Queue()
        self.__processed_images = []

    def provide(self, angle):
        pass

    def __obtain_alfas(self):
        if not constants.TRAIN_MPB:
            self.__alfas_from_file()
        else:
            self.__train()

    def __alfas_from_file(self):
        try:
            self.alfas = numpy.load('alfas.npy')
        except IOError:
            print('Sir no alfas found, proceeding to tame the beast')
            self.__train()

    def __train(self):
        truths_dir = constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_TRUTHS
        imgs_dir = constants.BSDS_PATH + constants.BSDS_PATH_TRAINING_IMGS
        truths_names = os.listdir(truths_dir)
        imgs_names = os.listdir(imgs_dir)
        for i in range(0, 3):
            img = cv2.imread(imgs_dir+imgs_names[i])

        numpy.save('alfas.npy', self.alfas)
