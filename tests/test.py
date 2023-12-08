import unittest
import calcfg
import numpy as np
from optimization_methods import ralgb5

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