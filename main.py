import numpy as np
from optimization_methods import ralgb5
from calcfg import calcfg2
from mpi4py import MPI

def main():
    number_of_start = 1000
    dim = 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    
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
    x_min = resultX_part[index_min]
    f_min = resultF_part[index_min]
    if(rank == 0):
        results = np.ones((numprocs, dim))
        results[0] = x_min[:]
        for k in range(1, numprocs):
            comm.Recv([x_min, dim, MPI.DOUBLE], source=k, tag=0, status=None)
            results[k] = x_min[:]
    else:
        comm.Send([x_min, dim, MPI.DOUBLE], dest=0, tag=0)

    if rank == 0:
        print(results, sep= "\n")

    

    
    


if __name__ == "__main__":
    main()