// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage
// Multiplication of elements in float array
//
// ***********************************************************************

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Demo kernel for array elements multiplication.
// Every thread selects one element and multiply it. 
__global__ void kernel_mult( float *pole, int L, float Mult )
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	// if grid is greater then length of array...
	if ( l >= L ) return;

	pole[ l ] *= Mult;
}

void run_mult( float *P, int Length, float Mult )
{
	cudaError_t cerr;
	int threads = 128;
	int blocks = ( Length + threads - 1 ) / threads;

	// Memory allocation in GPU device
	float *cudaP;
	cerr = cudaMalloc( &cudaP, Length * sizeof( float ) );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Copy data from PC to GPU device
	cerr = cudaMemcpy( cudaP, P, Length * sizeof( float ), cudaMemcpyHostToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Grid creation
	kernel_mult<<< blocks, threads >>>( cudaP, Length, Mult );

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from GPU device to PC
	cerr = cudaMemcpy( P, cudaP, Length * sizeof( float ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Free memory
	cudaFree( cudaP );
}