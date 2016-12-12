// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Image creation and its modification using CUDA.
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "pic_type.h"

using namespace cv;

// Prototype of function in .cu file
void run_animation( CUDA_Pic pic, uint2 block_size );

// Image size
#define SIZEX 432 // Width of image
#define	SIZEY 321 // Heigth of image
// Block size for threads
#define BLOCKX 40 // block width
#define BLOCKY 25 // block height

int main()
{
	// Array is created to store all points from image with size SIZEX * SIZEY. 
	// Image is stored line by line.
	// Creation of empty image
	Mat cv_img( SIZEY, SIZEX, CV_8UC3 );

	CUDA_Pic pic;
	pic.Size.x = SIZEX;
	pic.Size.y = SIZEY; 
	pic.PData = new uchar4[ pic.Size.x * pic.Size.y ];

	// Image filling by color gradient blue-green-red
	for ( int y = 0; y < pic.Size.y; y++ )
		for ( int x  = 0; x < pic.Size.x; x++ )
		{
			uchar4 bgr = { 0, 0, 0 }; // black
			if ( x < pic.Size.x / 2 )
			{
				bgr.y = 255 * x / ( pic.Size.x / 2 );
				bgr.x = 255 - bgr.y;
			}
			else
			{
				bgr.y = 255 * ( pic.Size.x - x ) / ( pic.Size.x / 2 );
				bgr.z = 255 - bgr.y;
			}
			// store points to array for transfer to GPU device
			pic.PData[ y * pic.Size.x + x ] = bgr;

			// store points to image
			Vec3b v3bgr( bgr.x, bgr.y, bgr.z );
			cv_img.at<Vec3b>( y, x ) = v3bgr;
		}

	// Show image before modification
	imshow( "B-G-R Gradient", cv_img );

	// Function calling from .cu file
	uint2 block_size = { BLOCKX, BLOCKY };
	run_animation( pic, block_size );

	// Store modified data to image
	for ( int y = 0; y < pic.Size.y; y++ )
		for ( int x  = 0; x < pic.Size.x; x++ )
		{
			uchar4 bgr = pic.PData[ y * pic.Size.x + x ];
			Vec3b v3bgr( bgr.x, bgr.y, bgr.z );
			cv_img.at<Vec3b>( y, x ) = v3bgr;
		}

	// Show modified image
	imshow( "B-G-R Gradient & Color Rotation", cv_img );
	waitKey( 0 );
}

