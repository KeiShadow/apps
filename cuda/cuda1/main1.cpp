// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
// Using global variables in threads, using printf.
//
// ***********************************************************************

#include <stdio.h>

// Prototype of function from .cu file
void run_cuda();

int main()
{
	// Function calling
	run_cuda();
	return 0;
}

