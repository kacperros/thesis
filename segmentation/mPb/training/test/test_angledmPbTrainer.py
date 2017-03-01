from unittest import TestCase

from segmentation.mPb.training.AngledmPbTrainer import AngledmPbTrainer


class TestAngledmPbTrainer(TestCase):
    def test_prepare(self):
        test_object = AngledmPbTrainer(90)
        test_object.prepare()
        print('Done')
