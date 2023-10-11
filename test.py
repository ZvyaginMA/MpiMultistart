import unittest
import calcfg
import numpy as np
from optimization_methods import ralgb5

class TestCalcfg(unittest.TestCase):
    def testCalcfg(self):
        input_data = np.array([3.0, 4.0])
        f, g = calcfg.calcfg(input_data)
        self.assertTrue(abs(f - 25) < 1E-10)
        self.assertTrue(abs(g - 2 * input_data).sum() < 1E-10)

class TestRalgb(unittest.TestCase):
    def test_ralgb_on_calcfg(self):
        x_start = np.array([4.0, 2.0])
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg.calcfg, x_start)
        self.assertTrue(abs(xr).sum() < 1E-10)
        self.assertTrue(abs(fr).sum() < 1E-10)