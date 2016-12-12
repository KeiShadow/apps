// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Image transformation from RGB to BW schema.
// Image manipulation is performed by OpenCV library.
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "pic_type.h"

using namespace cv;

// Function prototype from .cu file
void run_vlneni(int posun, int index );
void allocate(CUDA_Pic colorPic, CUDA_Pic bwPic);
void deallocate();

int main( int numarg, char **arg )
{
    int posun = 25;
    int index = 0;
    if ( numarg < 2 )
    {
        printf( "Enter picture filename!\n" );
        return 1;
    }

    // Load image
    Mat bgr_img = imread( arg[ 1 ], CV_LOAD_IMAGE_COLOR );
    imshow( "Original", bgr_img );

    CUDA_Pic bgr_pic, bw_pic;
    /*Originalni obrazek*/
    bgr_pic.Size.x = bgr_img.size().width;
    bgr_pic.Size.y = bgr_img.size().height;
    /*Vlneny obrazek*/
    bw_pic.Size.x = bgr_img.size().width; /*Sirka*/
    bw_pic.Size.y = bgr_img.size().height+2*posun; /*Vyska*/

    // Arrays alocation for images
    bgr_pic.PData = new uchar4[ bgr_pic.Size.x * bgr_pic.Size.y ];
    bw_pic.PData = new uchar4[ bw_pic.Size.x * bw_pic.Size.y ];
    for ( int y = 0; y < bgr_pic.Size.y; y++ )
        for ( int x  = 0; x < bgr_pic.Size.x; x++ )
        {
            Vec3b v3 = bgr_img.at<Vec3b>( y, x );
            uchar4 bgr = {  v3[ 0 ], v3[ 1 ], v3[ 2 ] };
            bgr_pic.PData[ y * bgr_pic.Size.x + x ] = bgr;

        }


    Mat bw_img( bw_pic.Size.y, bw_pic.Size.x, CV_8UC3 );
    allocate( bgr_pic, bw_pic);
    for(int q = 0; q < 36000; q++){
        run_vlneni(posun, index);

        for ( int y = 0; y < bw_pic.Size.y; y++ )
                for ( int x  = 0; x < bw_pic.Size.x; x++ )
                {
                    uchar4 bgr = bw_pic.PData[ x * bw_pic.Size.y+ y ];
                    Vec3b v3( bgr.x, bgr.y, bgr.z );
                    bw_img.at<Vec3b>( y, x ) = v3;
                }
        imshow( "Vlneni", bw_img );
        waitKey( 10 );
        index++;
    }
    deallocate();

    waitKey( 0 );
}
