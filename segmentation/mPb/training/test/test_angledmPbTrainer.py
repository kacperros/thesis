from time import time
from unittest import TestCase

from segmentation.mPb.training.AngledmPbTrainer import AngledmPbTrainer


class TestAngledmPbTrainer(TestCase):
    def test_prepare(self):
        time_s = time()
        test_object = AngledmPbTrainer(90)
        test_object.prepare()
        print(time() - time_s)
