#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <mpi.h>
#include <omp.h>

#include "Parallel_Matrix.hpp"
#include "GetPot"

using namespace Eigen;
using timer = std::chrono::high_resolution_clock;

int main(int argc, char **argv)
{	
	MPI_Init(&argc, &argv);

	// Set rank and number of processors
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Get the problem parameters in input
	GetPot cl(argc, argv);
	const int n = cl("dim", 10); 
	const unsigned int max_iter = cl("maxiter", 100000);
	const double tol = cl("tol", 1e-1);
	const int threads = cl("threads", 2);

	if(size > n)
	{
		std::cout << "Sorry, too many processors for this problem" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if(rank == 0)
    {
    	std::cout << "Number of MPI processors used: " << size << std::endl;
    	std::cout << "Number of OpenMP threads used: " << threads << std::endl;
    	std::cout << "Dimension of the problem: " << n << std::endl;
    }
    
    // BUild the local matrix
	Parallel_Matrix U(n, rank, size, threads);

	U.set_boundary_conditions("left", 1.0);
    
    U.set_boundary_conditions("right", 1.0);

    if(rank == 0)
    	U.set_boundary_conditions("top", 10.0);
    
    if(rank == size - 1)
    	U.set_boundary_conditions("bottom", 1.0);

    // Plot the initial global matrix (if it is not so big)
    Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> result;
	U.Gather_to_master(result);
	if(rank == 0 && n <= 20)
		std::cout << "Initial Matrix\n" << result << std::endl;

	// Start timer
	auto start = timer::now();
	
	// Apply the algorithm
	bool convergence = U.Jacobi(tol, max_iter);

	// End timer
	auto end = timer::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	
	// Print elapsed time and results
	if(rank == 0)
	{
		std::cout << "Elapsed Time: " << elapsed_time << " microseconds" << std::endl;
		if(convergence)
			std::cout << "Convergence reached and result saved into output_data file" << std::endl;
		else
			std::cout << "Algorithm does not converge, result at the last iteration saved into output_file" << std::endl;
	}
	
	// Gather the global results and save it in a file
	U.Gather_to_master(result);
	if(rank == 0)
	{	
		// print the result if the dimension is not too big
		if(n <= 20)
			std::cout << "Final result\n" << result << std::endl;

		// Save the result in a file
		std::ofstream file("output_data.dat");
		file << "Solution\n";
		for(int i = 0u; i < n; ++i)
		{
			for(int j = 0u; j < n; ++j)
			{
				file.setf(std::ios::left, std::ios::adjustfield);
				file.width(10);
				file << result(i,j);
			}

			file << "\n";
		}
		
		// Save the grid in a file
		file << "\nx grid\n";
		VectorXd Grid = VectorXd::LinSpaced(n, 0.0, 1.0);
		for(int i = 0u; i < n; ++i)
		{
			file.setf(std::ios::left, std::ios::adjustfield);
			file.width(10);
			file << Grid(i);
		}
		file << "\n";
		
		file << "\ny grid\n";
		for(int i = 0u; i < n; ++i)
		{	
			file.setf(std::ios::left, std::ios::adjustfield);
			file.width(10);
			file << Grid(i);
		}

		file.close();

	}

	MPI_Finalize();

	return 0;
}
