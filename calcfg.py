import numpy as np

def calcfg(x):
    """
        f(x) = x @ x
        g(x) = 2 * x
    """
    return x @ x , 2 * x

def calcfg2(x):
    """
        f(x) = x @ x
        g(x) = 2 * x
    """
    A = 10
    n = len(x)
    return A * n + (x * x - A * np.cos(x)).sum()


def calcfgTol(x):
    a, b = x[:len(x) // 2], x[len(x) // 2:]
    return - self.tol.tol_value(a, b), np.concatenate(
    [- self.tol.dTda(a, b), - self.tol.dTdb(a, b)])