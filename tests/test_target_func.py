import unittest
import calcfg
import numpy as np
from optimization_methods import ralgb5

class TestCalcfg(unittest.TestCase):
    def test_calcfg1(self):
        input_data = np.array([3.0, 4.0])
        f, g = calcfg.calcfg1(input_data)
        self.assertTrue(abs(f - 25) < 1E-10)
        self.assertTrue(abs(g - 2 * input_data).sum() < 1E-10)

    def test_calcfg2(self):
        input_data1 = np.array([0.0])
        f, g = calcfg.calcfg2(input_data1)
        self.assertTrue(abs(f) < 1E-10)
        self.assertTrue(abs(g).sum() < 1E-10)
        input_data2 = np.array([-2.465])
        f, g = calcfg.calcfg2(input_data2)
        self.assertTrue(np.abs(4.819 -f).sum() < 1E-2)

    def test_calcfg3(self):
        input_data1 = np.array([0.0, 1.0])
        f, g = calcfg.calcfg3(input_data1)
        self.assertTrue(np.abs(f - 1) < 1E-2)
        self.assertTrue(np.abs(np.array([0.0, 2.0]) - g).sum() < 1E-2)

        input_data2 = np.array([0.0, 0.0, 0.0])
        f, g = calcfg.calcfg3(input_data2)
        self.assertTrue(np.abs(f - 0) < 1E-2)
        self.assertTrue(np.abs(np.array([0.0, 0.0, 0.0]) - g).sum() < 1E-2)