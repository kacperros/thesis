import unittest

from segmentation.mPb.AngledmPbCalculatorProvider import AngledmPbCalculatorProvider


class AngledmPbCalculatorProviderTest(unittest.TestCase):
    def test_something(self):
        tested_object = AngledmPbCalculatorProvider()
        provided_obj = tested_object.provide(45)


if __name__ == '__main__':
    unittest.main()
