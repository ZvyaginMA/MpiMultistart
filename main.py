import numpy as np
from optimization_methods import ralgb5
from calcfg import calcfg2
from mpi4py import MPI
from datetime import datetime

class MultistartOptions:
    number_of_start : int
    dim : int
    l_boundary : np.ndarray 
    u_boundary : np.ndarray

def parallel_multistart(calc_fg, method_optimization, multistart_options : MultistartOptions):
    dim = multistart_options.dim
    number_of_start = multistart_options.number_of_start
    lb = multistart_options.l_boundary
    ub = multistart_options.u_boundary

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    if(rank == 0):
        t1 = datetime.now()
    
    X_part = np.zeros((number_of_start // numprocs , dim), dtype=np.float64)
    for i in range(number_of_start // numprocs):
        X_part[i] = np.random.uniform(lb, ub)

    resultX_part = np.zeros((number_of_start // numprocs, dim))
    resultF_part = np.zeros((number_of_start // numprocs, 1))
    for i in range(len(X_part)):
        xr, fr, nit, ncalls, ccode= method_optimization(calc_fg, X_part[i])
        resultX_part[i] = xr
        resultF_part[i] = fr

    index_min = np.argmin(resultF_part)
    res_part = np.zeros((dim + 1), dtype=np.float64)
    res_part[:-1] = resultX_part[index_min][:]
    res_part[-1] = np.float64(resultF_part[index_min])
    status = MPI.Status()
    if(rank == 0):
        results = np.ones((numprocs, dim + 1))
        results[0] = res_part[:-1]
        results[0][-1] = res_part[-1]
        for k in range(1, numprocs):
            comm.Probe(source=MPI.ANY_SOURCE , tag=MPI.ANY_TAG , status=status) #ловим первый попавшийся send и из 
            comm.Recv([results[status.Get_source()], dim + 1, MPI.DOUBLE], source=status.Get_source(), tag=0, status=None)
    else:
        comm.Send([res_part, dim + 1, MPI.DOUBLE], dest=0, tag=0)

    if rank == 0:
        index_min = np.argmin(results, axis=0)[-1]
        x_total_min = results[index_min][:-1]
        f_total_min = results[index_min][-1]
        print(f"Time {datetime.now() - t1}")
        print(f"Multistart found a solution to x = {x_total_min} and f = {f_total_min} in {number_of_start} iterations")
    

if __name__ == "__main__":
    m_opt = MultistartOptions()
    m_opt.dim = 1
    m_opt.u_boundary = np.ones((m_opt.dim)) * 500.0
    m_opt.l_boundary = - np.ones((m_opt.dim)) * 500.0
    m_opt.number_of_start = 1000
    parallel_multistart(calcfg2, ralgb5, m_opt)