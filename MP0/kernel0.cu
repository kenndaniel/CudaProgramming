
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include <ctime>

#include <cstdio>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>  
#include	<vector>
#include	<string>

#define wbCheck(stmt)  do { cudaError_t err = stmt; if (err != cudaSuccess) {printf( "Failed to run stmt ", stmt); return -1;}} while(0)    


#define GPU 5

//#include	<wb.h>

	 int whatDevice()
	 { 
	int deviceCount;

    //wbArg_read(argc, argv);

    wbCheck(cudaGetDeviceCount(&deviceCount));

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
                printf( "No CUDA GPU has been detected \n");
                return -1;
            } else if (deviceCount == 1) {
                printf( "There is 1 device supporting CUDA \n");
            } else {
                printf( "There are %d devices supporting CUDA",deviceCount);
            }
			
        }

        printf( "Device %d name: %s \n", dev,deviceProp.name);
        printf( " Computational Capabilities: %d . %d \n", deviceProp.major,  deviceProp.minor);
        printf( " Maximum global memory size: %d \n", deviceProp.totalGlobalMem);
        printf(" Maximum constant memory size: %u \n", deviceProp.totalConstMem);
        printf( " Maximum shared memory size per block: %d \n", deviceProp.sharedMemPerBlock);
		printf( " Maximum threads per block: %d \n",1, deviceProp.maxThreadsPerBlock);
        		printf(" %s %d x %d x %d \n"," Maximum block dimensions: ", deviceProp.maxThreadsDim[0],
                                                    deviceProp.maxThreadsDim[1],
                                                    deviceProp.maxThreadsDim[2]);

        printf(" %s %d x %d x %d \n", " Maximum grid dimensions: ", deviceProp.maxGridSize[0],
                                                   deviceProp.maxGridSize[1],
                                                   deviceProp.maxGridSize[2]);
         printf(" Warp size: %d \n", deviceProp.warpSize);
    }
	return 1;
	 }

clock_t start;
double diff;
void wbTime_start(char *msg) 
	{
		printf(msg); 
		start = clock();
	}

void wbTime_stop(char* msg)
	{

		(double)diff = ( clock() - start ) / (double)CLOCKS_PER_SEC;
		
		printf("%s   %f \n",msg,diff);
}

//@@ The purpose of this code is to become familiar with the submission 
//@@ process. Do not worry if you do not understand all the details of 
//@@ the code.

int main(int argc, char ** argv) {

	wbTime_start(" Start");
		 whatDevice();

    wbTime_stop( "Getting GPU Data."); //@@ stop the timer
	    wbCheck( cudaDeviceReset());
    
    return 1;
}