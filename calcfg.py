import numpy as np

def calcfg1(x):
    """
        f(x) = x @ x
        g(x) = 2 * x
    """
    return x @ x , 2 * x

def calcfg2(x):
    """
        f(x) = x^2 * cos(14x) + 1.7x^2
        g(x) = 2x * cos(14x) - 14 *x^2 * sin(14x) + 3.4x
    """
    if (len(x) > 1):
        raise Exception(f"x must have len = 1 but have {len(x)}")
    f = x*x * np.cos(4 * x) + 1.7 * x * x
    g = 2 * x * np.cos(14 * x) - 14 *x * x * np.sin(14 * x) + 3.4 * x
    return f, g


def calcfgTol(x):
    a, b = x[:len(x) // 2], x[len(x) // 2:]
    return - self.tol.tol_value(a, b), np.concatenate(
    [- self.tol.dTda(a, b), - self.tol.dTdb(a, b)])