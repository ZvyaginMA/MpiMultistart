import unittest
import calcfg
import numpy as np
from optimization_methods import ralgb5

class TestRalgb(unittest.TestCase):
    def test_ralgb_on_calcfg1(self):
        x_start = np.array([4.0, 2.0])
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg.calcfg1, x_start)
        self.assertTrue(abs(xr).sum() < 1E-10)
        self.assertTrue(abs(fr).sum() < 1E-10)

    def test_ralgb_on_calcfg2(self):
        x_start = np.array([4.0])
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg.calcfg2, x_start)
        self.assertTrue(abs(xr - 2.37826).sum() < 1E-4)
        # нашли локальный минимум