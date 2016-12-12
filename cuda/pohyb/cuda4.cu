// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Image transformation from RGB to BW schema.
//
// ***********************************************************************

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "pic_type.h"
//#include "math.h"

CUDA_Pic cudaColorPic;
CUDA_Pic cudaBWPic;
CUDA_Pic colorPic;
CUDA_Pic bwPic;

// Demo kernel to tranfrom RGB color schema to BW schema
__global__ void kernel_vlneni( CUDA_Pic colorPic, CUDA_Pic bwPic, int posun, int index )
{

    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if ( y >= colorPic.Size.y ) return;
    if ( x >= colorPic.Size.x ) return;


    

    uchar4 bgr = colorPic.PData[ y * colorPic.Size.x+x];
    //int position = x+index*posun;
    //int finpos = x*bwPic.Size.y + y+position;
    

    //float sinus = sin((float)(x+index)/posun);
    // float position = (x + posun)*bwPic.Size.y+y;
    //float position =  x* bwPic.Size.y + y + (posun + posun*sinus);
    bwPic.PData[y*bwPic.Size.x+(x+index*posun)]= bgr;


}


void run_vlneni(int posun, int index )
{

    cudaError_t cerr;
    int block_size = 16;

    dim3 blocks( ( cudaColorPic.Size.x + block_size-1  ) / block_size, ( cudaColorPic.Size.y + block_size-1  ) / block_size );
    dim3 threads( block_size, block_size );

    cudaMemset(cudaBWPic.PData, 0, cudaBWPic.Size.x * cudaBWPic.Size.y * sizeof( uchar4 ));

    // Grid creation, size of grid must be greater than image
    kernel_vlneni<<< blocks, threads >>>( cudaColorPic, cudaBWPic, posun, index );

    if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

    // Copy new image from GPU device
    cerr = cudaMemcpy( bwPic.PData, cudaBWPic.PData, bwPic.Size.x * bwPic.Size.y * sizeof( uchar4 ), cudaMemcpyDeviceToHost );
    if ( cerr != cudaSuccess )
        printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

    /*cerr = cudaMemcpy( colorPic.PData, cudaColorPic.PData, colorPic.Size.x * colorPic.Size.y * sizeof( uchar4 ), cudaMemcpyDeviceToHost );
        if ( cerr != cudaSuccess )
            printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );*/


}

void allocate(CUDA_Pic colorPic2, CUDA_Pic bwPic2)
{
    cudaError_t cerr;
        // Memory allocation in GPU device
        colorPic = colorPic2;
        bwPic = bwPic2;
        cudaColorPic.Size = colorPic.Size;
        cudaBWPic.Size = bwPic.Size;

        cerr = cudaMalloc( &cudaColorPic.PData, cudaColorPic.Size.x * cudaColorPic.Size.y * sizeof( uchar4 ) );
        if ( cerr != cudaSuccess )
            printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

        cerr = cudaMalloc( &cudaBWPic.PData, cudaBWPic.Size.x * cudaBWPic.Size.y * sizeof( uchar4 ) );
        if ( cerr != cudaSuccess )
            printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

       // cudaMemset(cudaBWPic.PData, 0, cudaBWPic.Size.x * cudaBWPic.Size.y * sizeof( uchar4 ));

        // Copy color image to GPU device
        cerr = cudaMemcpy( cudaColorPic.PData, colorPic.PData, cudaColorPic.Size.x * cudaColorPic.Size.y * sizeof( uchar4 ), cudaMemcpyHostToDevice );
        if ( cerr != cudaSuccess )
            printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );
}

void deallocate()
{
    // Free memory
        cudaFree( cudaColorPic.PData );/*Uvolneni puvodniho obrazku*/
        cudaFree( cudaBWPic.PData );/*Uvolneni deformovaneho obrazku*/
}
