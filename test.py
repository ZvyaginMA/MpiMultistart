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
        f, g = calcfg.calcfg1(input_data1)
        self.assertTrue(abs(f) < 1E-10)
        self.assertTrue(abs(g).sum() < 1E-10)
        input_data2 = np.array([-2.318])
        f, g = calcfg.calcfg2(input_data2)
        self.assertTrue(np.abs(3.8237 -f).sum() < 1E-3)
        self.assertTrue(abs(g).sum() < 1E-1)

class TestRalgb(unittest.TestCase):
    def test_ralgb_on_calcfg1(self):
        x_start = np.array([4.0, 2.0])
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg.calcfg1, x_start)
        self.assertTrue(abs(xr).sum() < 1E-10)
        self.assertTrue(abs(fr).sum() < 1E-10)

    def test_ralgb_on_calcfg2(self):
        x_start = np.array([4.0])
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg.calcfg2, x_start)

        self.assertTrue(abs(xr - 2.31764835).sum() < 1E-5)
        self.assertTrue(abs(fr - 3.82376726).sum() < 1E-5)
        # нашли локальный минимум