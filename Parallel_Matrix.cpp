#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
#include <omp.h>

#include "Parallel_Matrix.hpp"

void Parallel_Matrix::set_boundary_conditions(const std::string& patch, const double& value)
{
	if(patch == "top")
	{
		if (rank == 0)
			values.row(0) = ArrayXd::Constant(values.cols(), value);
		else
			std::cout << "boundary condition on top available only for rank 0" << std::endl;
	}
		
	if(patch == "bottom")
	{
		if(rank == size - 1)
			values.row(values.rows() - 1) = ArrayXd::Constant(values.cols(), value);
		else
			std::cout << "boundary condition on bottom available only for the last rank" << std::endl;

	}

	if(patch == "right")
		values.col(values.cols() - 1) = ArrayXd::Constant(values.rows(), value);

	if(patch == "left")
		values.col(0) = ArrayXd::Constant(values.rows(), value);

	return;
}

bool Parallel_Matrix::Jacobi(const double& tol, const unsigned int& max_iter)
{
	unsigned int iter = 0u;
	bool stop = false;
	bool all_stop = false;
	double increment;
	Matrix<double, Dynamic, Dynamic, RowMajor> values_old = values;

	VectorXd tmp1(values.cols());
	VectorXd tmp2(values.cols());

	while(iter < max_iter && !all_stop)
	{
		increment = 0.0;

		// Serial case
		if(size == 1)
		{	
			#pragma omp parallel for num_threads(threads) collapse(2) reduction (+:increment)
			for(unsigned int i = 1u; i < values.rows() - 1; ++i)
			{
				for(unsigned int j = 1u; j < values.cols() - 1; ++j)
				{	
					values_old(i,j) = values(i,j);

					values(i,j) = 0.25*(values_old(i+1,j) + values_old(i-1,j) + values_old(i,j+1) + values_old(i,j-1));

					increment += (values(i,j) - values_old(i,j))*(values(i,j) - values_old(i,j));
				}
			}

			all_stop = std::sqrt(increment) < tol;
		}

		// Parallel case
		else
		{	
			// Communications among processors

			if(rank != 0 && rank != size - 1)
			{
				MPI_Status status;
				MPI_Sendrecv(values.row(0).data(), values.cols(), MPI_DOUBLE, rank - 1, 100*rank, tmp1.data(), values.cols(), MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &status);
				MPI_Sendrecv(values.row(values.rows() - 1).data(), values.cols(), MPI_DOUBLE, rank + 1, rank + 1, tmp2.data(), values.cols(), MPI_DOUBLE, rank + 1, 100*(rank + 1), MPI_COMM_WORLD, &status);
			}

			if(rank == 0)
			{
				MPI_Status status;
				MPI_Sendrecv(values.row(values.rows() - 1).data(), values.cols(), MPI_DOUBLE, rank + 1, rank + 1, tmp2.data(), values.cols(), MPI_DOUBLE, rank + 1, 100*(rank + 1), MPI_COMM_WORLD, &status);
			}

			if(rank == size - 1)
			{
				MPI_Status status;
				MPI_Sendrecv(values.row(0).data(), values.cols(), MPI_DOUBLE, rank - 1, 100*rank, tmp1.data(), values.cols(), MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &status);
			}
		

			// Update step
			#pragma omp parallel for num_threads(threads) reduction (+:increment)
			for(unsigned int i = 0u; i < values.rows(); ++i)
			{	
				// Skip the boundaries that are fixed
				if(i == 0u && rank == 0)
					continue;
				else if (i == values.rows() - 1 && rank == size - 1)
					continue;

				// Case with 1 row only
				else if(values.rows() == 1 && rank != 0 && rank != size - 1)
				{
					for(unsigned int j = 1u; j < values.cols()-1; ++j)
					{	
						values_old(i,j) = values(i,j);

						values(i,j) = 0.25*(tmp1(j) + tmp2(j) + values_old(i,j+1) + values_old(i,j-1));

						increment += (values(i,j) - values_old(i,j))*(values(i,j) - values_old(i,j));
					}
				}

				// Update for the first line
				else if(i == 0u && rank != 0)
				{
					for(unsigned int j = 1u; j < values.cols()-1; ++j)
					{	
						values_old(i,j) = values(i,j);

						values(i,j) = 0.25*(tmp1(j) + values_old(i+1,j) + values_old(i,j+1) + values_old(i,j-1));

						increment += (values(i,j) - values_old(i,j))*(values(i,j) - values_old(i,j));
					}
				}


				// Update for the last line
				else if(i == values.rows() - 1 && rank != size - 1)
				{	
					for(unsigned int j = 1u; j < values.cols()-1; ++j)
					{	
						values_old(i,j) = values(i,j);

						values(i,j) = 0.25*(tmp2(j) + values_old(i-1,j) + values_old(i,j+1) + values_old(i,j-1));

						increment += (values(i,j) - values_old(i,j))*(values(i,j) - values_old(i,j));
					}
				}

				else
				{
					for(unsigned int j = 1u; j < values.cols()-1; ++j)
					{	
						values_old(i,j) = values(i,j);

						values(i,j) = 0.25*(values_old(i+1,j) + values_old(i-1,j) + values_old(i,j+1) + values_old(i,j-1));

						increment += (values(i,j) - values_old(i,j))*(values(i,j) - values_old(i,j));
					}
				}
				
			}

			stop = std::sqrt(increment) < tol;

			MPI_Allreduce(&stop, &all_stop, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

		}

		iter++;
	}

	return iter < max_iter;

}

void Parallel_Matrix::Gather_to_master(Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>& result) const
{	
	if(size == 1)
	{
		result = values;
		return;
	}

	// Vectors to store the number of elements to gather from each
	// processor and the offset index where to start writing them into.
	std::vector<int> recv_counts;
	std::vector<int> recv_start_idx;
	recv_counts.resize(size);
	recv_start_idx.resize(size);

	int start_idx = 0;
	for (int i = 0; i < size; ++i)
	{
		recv_counts[i] = (i < values.cols()%size) ? (values.cols()/size + 1) : values.cols()/size;
		recv_counts[i] *= values.cols();

		recv_start_idx[i] = start_idx;

		start_idx += recv_counts[i];
	}

	if(rank == 0)
		result.resize(values.cols(), values.cols());
	
	MPI_Gatherv(values.data(),
				values.cols()*values.rows(),
				MPI_DOUBLE,
				result.data(),
				recv_counts.data(),
				recv_start_idx.data(),
				MPI_DOUBLE,
				0,
				MPI_COMM_WORLD);
}
