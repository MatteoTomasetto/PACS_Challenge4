#ifndef PARALLEL_MATRIX_HPP_
#define PARALLEL_MATRIX_HPP_

#include <Eigen/Dense>
#include <string>
#include <vector>

using namespace Eigen;

class Parallel_Matrix
{
public:
	// Matrix values
	Matrix<double, Dynamic, Dynamic, RowMajor> values;
	
	// Rank of the processor
	const int rank;
	
	// Number of MPI processors used
	const int size;

	// Number of OpenMP threads used
	const int threads;


	// Constructor that takes the rank in input and builds the local matrix (filled with zeros)
	Parallel_Matrix(const int& n, const int& rank_, const int& size_, const int& threads_)
	: rank(rank_), size(size_), threads(threads_)
	{
		int n_rows = n / size;
		
		if(rank < (n % size))
			n_rows = n_rows + 1;
		
		values.resize(n_rows, n);

		values = MatrixXd::Zero(n_rows, n);
	}

	// Add boundary conditions to the matrix
	void set_boundary_conditions(const std::string& patch, const double& value);

	// Jacobi iteration method
	bool Jacobi(const double& tol, const unsigned int& max_iter);

	// Gather the local matrices in "result" matrix of rank 0
	void Gather_to_master(Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>& result) const;

};

#endif
