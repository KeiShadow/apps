// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Image interface for CUDA
//
// ***********************************************************************


// Structure definition for exchanging data between Host and Device
struct CUDA_Pic
{
  uint2 Size;			// size of picture
  uchar4 *PData;		// data of picture
};
