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


def calcfg3(x):
    """
        f(x) = 20 + x^2 - 10*cos(2pi x) + y^2 - 10*cos(2pi y)
        g(x) = (2x + 20pi sin(2pix),  2y + 20pi sin(2pi y))

        экстремум (x, y) = (0, 0)
        https://ru.wikipedia.org/wiki/%D0%A4%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D1%8F_%D0%A0%D0%B0%D1%81%D1%82%D1%80%D0%B8%D0%B3%D0%B8%D0%BD%D0%B0
    """
    f = 10 * len(x) + sum(x * x) - 10 * sum(np.cos(2 * np.pi * x)) 
    g = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x) 
    return f, g

