// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Threads organization in grid.
// Every thread displays information of its position in block and 
// position of block in grid. 
//
// ***********************************************************************


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Demo kernel will display all global variables of grid organization.
// Warning! Function printf is available from compute capability 2.x
__global__ void thread_hierarchy()
{
    // Global variables
    // Grid dimension -				gridDim
	// Block position in grid -		blockIdx
	// Block dimension -			blockDim
	// Thread position in block -	threadIdx
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    printf( "Block{%d,%d}[%d,%d] Thread{%d,%d}[%d,%d] [%d,%d]\n",
	    gridDim.x, gridDim.y, blockIdx.x, blockIdx.y,
		blockDim.x, blockDim.y, threadIdx.x, threadIdx.y, x, y );
}

void run_cuda()
{
	cudaError_t cerr;
	// Following command can increase internal buffer for printf function
    /*cerr = cudaDeviceSetLimit( cudaLimitPrintfFifoSize, required_size );
	if ( err != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );
    */

	// Thread creation from selected kernel:
	// first parameter dim3 is grid dimension
	// second parameter dim3 is block dimension
    thread_hierarchy<<< dim3( 3, 2 ), dim3( 2, 3 )>>>();

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Output from printf is in GPU memory. 
	// To get its contens it is necessary to synchronize device.

	cudaDeviceSynchronize();
}
