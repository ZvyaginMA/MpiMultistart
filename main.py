import numpy as np
from optimization_methods import ralgb5
from calcfg import calcfg2
from mpi4py import MPI
from datetime import datetime

def main():
    number_of_start = 1000
    dim = 1
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    if(rank == 0):
        print(datetime.now())
    lb = np.array([-50.0])
    ub = np.array([50.0])
    X_part = np.zeros((number_of_start // numprocs , dim), dtype=np.float64)
    for i in range(number_of_start // numprocs):
        X_part[i] = np.random.uniform(lb, ub)

    resultX_part = np.zeros((number_of_start // numprocs, dim))
    resultF_part = np.zeros((number_of_start // numprocs, 1))
    for i in range(len(X_part)):
        xr, fr, nit, ncalls, ccode= ralgb5(calcfg2, X_part[i])
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
        for k in range(1, numprocs):
            comm.Probe(source=MPI.ANY_SOURCE , tag=MPI.ANY_TAG , status=status)
            comm.Recv([results[status.Get_source()], dim + 1, MPI.DOUBLE], source=status.Get_source(), tag=0, status=None)
    else:
        comm.Send([res_part, dim + 1, MPI.DOUBLE], dest=0, tag=0)

    if rank == 0:
        print(datetime.now())
        print(results, sep= "\n")
        index_min = np.argmin(results, axis=0)[-1]
        print(results[index_min][:-1])

    

    
    


if __name__ == "__main__":
    main()