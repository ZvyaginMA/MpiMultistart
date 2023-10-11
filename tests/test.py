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

class TestSeqMultistart(unittest.TestCase):
    def test1(self):
        number_of_start = 1000
        dim = 1
        lb = np.array([-50.0])
        ub = np.array([50.0])
        X = np.zeros((number_of_start, dim), dtype=np.float64)
        for i in range(number_of_start):
            X[i] = np.random.uniform(lb, ub)

        resultX = np.zeros((number_of_start, dim))
        resultF = np.zeros((number_of_start))
        for i in range(len(X)):
            xr, fr, nit, ncalls, ccode= ralgb5(calcfg.calcfg2, X[i])
            resultX[i] = xr
            resultF[i] = fr

        index_min = np.argmin(resultF)
        x_min = resultX[index_min]
        f_min = resultF[index_min]
        self.assertTrue(np.abs(x_min).sum() < 1E-5)
        self.assertTrue(np.abs(f_min).sum() < 1E-5)
        #Нашли глобальный минимум