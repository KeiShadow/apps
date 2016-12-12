// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Manipulation with prepared image.
//
// ***********************************************************************

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "pic_type.h"

// Every threads identifies its position in grid and in block and modify image
__global__ void kernel_animation( CUDA_Pic cuda_pic )
{
	// X,Y coordinates 
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if ( x >= cuda_pic.Size.x ) return;
	if ( y >= cuda_pic.Size.y ) return;

	// Point [x,y] selection from image
	uchar4 bgr = cuda_pic.PData[ y * cuda_pic.Size.x + x ];

        // Color rotation inside block
        int x2 = blockDim.x / 2;
        int y2 = blockDim.y / 2;
        int px = __sad( x2, threadIdx.x, 0 ); // abs function
        int py = __sad( y2, threadIdx.y, 0 );

        if ( px < x2 * ( y2 - py ) / y2 ) 
        {
                uchar4 tmp = bgr;
                bgr.x = tmp.y;
                bgr.y = tmp.z;
                bgr.z = tmp.x;
        }

	// Store point [x,y] back to image
	cuda_pic.PData[ y * cuda_pic.Size.x + x ] = bgr;

}

void run_animation( CUDA_Pic pic, uint2 block_size )
{
	cudaError_t cerr;

	CUDA_Pic cudaPic;
	cudaPic.Size = pic.Size;

	// Memory allocation in GPU device
	cerr = cudaMalloc( &cudaPic.PData, cudaPic.Size.x * cudaPic.Size.y * sizeof( uchar4 ) );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Copy data to GPU device
	cerr = cudaMemcpy( cudaPic.PData, pic.PData, cudaPic.Size.x * cudaPic.Size.y * sizeof( uchar4 ), cudaMemcpyHostToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Grid creation with computed organization
	dim3 mrizka( ( cudaPic.Size.x + block_size.x - 1 ) / block_size.x, ( cudaPic.Size.y + block_size.y - 1 ) / block_size.y );
	kernel_animation<<< mrizka, dim3( block_size.x, block_size.y ) >>>( cudaPic );

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from GPU device to PC
	cerr = cudaMemcpy( pic.PData, cudaPic.PData, pic.Size.x * pic.Size.y * sizeof( uchar4 ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );	

	// Free memory
	cudaFree( cudaPic.PData );

	// For printf
	//cudaDeviceSynchronize();

}
