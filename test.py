import unittest
import calcfg
import numpy as np

class TestCalcfg(unittest.TestCase):
    def test1(self):
        input_data = np.array([3.0, 4.0])
        f, g = calcfg.calcfg(input_data)
        self.assertTrue(abs(f - 25) < 1E-10)
        self.assertTrue(abs(g - 2 * input_data).sum() < 1E-10)

