/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>

#include <cublas_v2.h>
#include <mpi.h>

#include "common.h"

#define NY 4096
#define NX 4096

typedef real arrtype[NX];

__global__ void calculate_jacobi(arrtype *A, arrtype *rhs, arrtype *Anew, int width, int height, real *error){

  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if( (row > 0 && row < height-1) && (col > 0 && col < width-1) ){

    Anew[row][col] = -0.25 * (rhs[row][col] - ( A[row][col+1] + A[row][col-1]
                                            + A[row-1][col] + A[row+1][col] ));

		error[row*width + col] = fabs(Anew[row][col]-A[row][col]);

  }

}

__global__ void Anew_to_A(arrtype *A, arrtype *Anew, int width, int height){

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if( (row > 0 && row < height-1) && (col > 0 && col < width-1) ){
		A[row][col] = Anew[row][col];
	}	

}

__global__ void periodic_BC_rows(arrtype *A, int width, int height){

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if( (row == 0) && (col > 0 && col < width-1) ){
		A[row][col] = A[(height-2)][col];
	}

	if( (row == (height-1)) && (col > 0 && col < width-1) ){
    A[row][col] = A[1][col];
  }

}

__global__ void periodic_BC_columns(arrtype *A, int width, int height){

  int col = threadIdx.x + blockDim.x * blockIdx.x;
  int row = threadIdx.y + blockDim.y * blockIdx.y;

  if( (row > 0 && row < height-1) && (col == 0) ){
    A[row][col] = A[row][(width-2)];
  }

	if( (row > 0 && row < height-1) && (col == (width-1)) ){
    A[row][col] = A[row][1];
  }

}

int main(int argc, char* argv[])
{

	MPI_Init(&argc, &argv);

	int num_ranks;
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int num_devices;
	cudaGetDeviceCount(&num_devices);

	cudaSetDevice(rank);

	int iter_max   = 1000;
	const real tol = 1.0e-5;

	cublasHandle_t handle;

	cublasCreate(&handle);

	real A[NY][NX];
	real rhs[NY][NX];

	real error[NY][NX];


	// set A
	for(int i=0; i<NY; i++){
		for(int j=0; j<NX; j++){
			A[i][j] = 0.0;
			error[i][j] = 0.0;
		}
	}

	// set rhs
	for (int iy = 1; iy < NY-1; iy++)
	{
		for( int ix = 1; ix < NX-1; ix++ )
		{
			const real x = -1.0 + (2.0*ix/(NX-1));
			const real y = -1.0 + (2.0*iy/(NY-1));
			rhs[iy][ix] = expr(-10.0*(x*x + y*y));
		}
	}

	printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);

	StartTimer();

	arrtype *d_A;
	cudaMalloc(&d_A, NX*NY*sizeof(real));
	cudaMemcpy(d_A, A, NX*NY*sizeof(real), cudaMemcpyHostToDevice);

	arrtype *d_rhs;
	cudaMalloc(&d_rhs, NX*NY*sizeof(real));
	cudaMemcpy(d_rhs, rhs, NX*NY*sizeof(real), cudaMemcpyHostToDevice);

  arrtype *d_Anew;
	cudaMalloc(&d_Anew, NX*NY*sizeof(real));

	real *d_error;
  cudaMalloc(&d_error, NX*NY*sizeof(real));
	cudaMemcpy(d_error, error, NX*NY*sizeof(real), cudaMemcpyHostToDevice);

	dim3 threads_per_block(16,16,1);
	dim3 blocks_in_grid( ceil( real(NX) / threads_per_block.x ), ceil ( real(NY) / threads_per_block.y ), 1 );

	int iter   = 0;

	int max_index;
	real max_error = 1.0;

	while ( max_error > tol && iter < iter_max ){

		calculate_jacobi<<<blocks_in_grid, threads_per_block>>>(d_A, d_rhs, d_Anew, NX, NY, d_error);
		cudaDeviceSynchronize();

//		cublasIsamax(handle, NY*NX, d_error, 1, &max_index);
		cublasIdamax(handle, NY*NX, d_error, 1, &max_index);
		cudaMemcpy(&max_error, d_error+max_index-1, sizeof(real), cudaMemcpyDeviceToHost);

		Anew_to_A<<<blocks_in_grid, threads_per_block>>>(d_A, d_Anew, NX, NY);
		cudaDeviceSynchronize();

		periodic_BC_rows<<<blocks_in_grid, threads_per_block>>>(d_A, NX, NY);
		cudaDeviceSynchronize();

		periodic_BC_columns<<<blocks_in_grid, threads_per_block>>>(d_A, NX, NY);
		cudaDeviceSynchronize();

		if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, max_error);

		iter++;	

	}

	cudaMemcpy(A, d_A, NX*NY*sizeof(real), cudaMemcpyDeviceToHost);

	double runtime = GetTimer();

	printf( "%dx%d: GPU %d of %d: %8.4f s\n", NY,NX, rank, num_devices, runtime/ 1000.0 );

	cudaFree(d_A);
	cudaFree(d_rhs);
	cudaFree(d_Anew);
	cudaFree(d_error);

	MPI_Finalize();

  return 0;
}
