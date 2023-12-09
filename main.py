import numpy as np
from optimization_methods import ralgb5
from calcfg import calcfg2, calcfg3
from multistart_options import MultistartOptions
from parallel_multistart import parallel_multistart

def calc_calcfg2():
    m_opt = MultistartOptions()
    m_opt.dim = 1
    m_opt.u_boundary = np.ones((m_opt.dim)) * 500.0
    m_opt.l_boundary = - np.ones((m_opt.dim)) * 500.0
    m_opt.number_of_start = 1000
    parallel_multistart(calcfg2, ralgb5, m_opt)

def calc_calcfg3():
    m_opt = MultistartOptions()
    m_opt.dim = 4
    m_opt.u_boundary = np.ones(m_opt.dim) * 5.0
    m_opt.l_boundary = np.ones(m_opt.dim) * -5.0
    m_opt.number_of_start = 10000
    parallel_multistart(calcfg3, ralgb5, m_opt)

if __name__ == "__main__":
    calc_calcfg2()

