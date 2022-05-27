# Jacobi iteration method for the Laplace Equation #

This program computes the solution of the Laplace Equation with null forcing term in a squared domain (0,1)X(0,1) via Jacobi iteration method exploiting parallel computing techniques (MPI and OpenMP).

In particular the following boundary conditions are considered:
- u = 10 on the top of the domain;
- u = 1 on the bottom of the domain;
- u = 1 on the right side of the domain;
- u = 1 on the left side of the domain.

The following parameters are taken in input from command line thanks to GetPot:
- dim = number of grid points considered for each side of the domain;
- tol = tolerance for the Jacobi iteration method;
- maxiter = maximum number of iterations for the Jacobi iteration method;
- threads = number of OpenMP threads to use

Example of execution: `mpirun -n 4 ./main "dim"=100 "tol"=1e-1 "maxiter"=100 "threads"=4`

By default: dim = 10, tol=1e-1, maxiter=100000, threads=2.

Notice that you can set the number of MPI processors to use thank to the `-n` entry of mpirun.

The result is saved in a file called `output_data.dat`.
In addition, in this folder you can find a table to investigate the parallel computing speedup and a plot of the solution when dim=100, threads=1, maxiter=100000 and tol=1e-1. 

Notice that the advantages of parallelization are considerable as the dimension of the problem increases; instead the serial version is faster for small dimensions due to parallel overhead.

In this directory, `make` produces the executable which is just called `main`.
