
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include	<wb.h>
#include <stdio.h>
#include <stdlib.h>  
#define wbCheck(stmt)  do { cudaError_t err = stmt; if (err != cudaSuccess) {printf( "Failed to run stmt ", stmt); return -1;}} while(0)    
#define wbLog(level, msg)   printf(msg)

// MP 1


__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int i = blockIdx.x * blockDim.x+ threadIdx.x;
	
	if( i<len ) out[i] = in1[i]+in2[i];
}

int main(int argc, char ** argv) {
 //   wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;
	float * hostCheckOutput;

//    args = wbArg_read(argc, argv);

 //   wbTime_start(Generic, "Importing data and creating memory on host");
	inputLength = 1000000;
   hostInput1 = (float *)malloc(inputLength * sizeof(float));
   hostInput2 = (float *)malloc(inputLength * sizeof(float));
		hostOutput = (float *)malloc(inputLength * sizeof(float));
		hostCheckOutput= (float *)malloc(inputLength * sizeof(float));
		printf("1  %i", inputLength);
 //   wbTime_stop(Generic, "Importing data and creating memory on host");
		 for (int i = 0; i < inputLength; i++)
    {
        hostInput1[i] = rand() / (float)RAND_MAX;
        hostInput2[i] = rand() / (float)RAND_MAX;
		hostCheckOutput[i] = hostInput1[i] + hostInput2[i];
       
    }
		 printf("2");
 //   wbLog(TRACE, "The input length is ", inputLength);

//	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	int size = inputLength * sizeof(float); 
	

		wbCheck(cudaMalloc((void **) &deviceInput1, size));  // Input
		wbCheck(cudaMalloc((void **) &deviceInput2, size));  // Input
	
		wbCheck(cudaMalloc((void **) &deviceOutput, size));  // Output
		printf("3");
 //   wbTime_stop(GPU, "Allocating GPU memory.");

 //   wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here

		wbCheck(cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice));
		wbCheck(cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice));
	printf("4");
 //   wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
		int blockSize = 1024;
		int gridSize =(inputLength-1)/blockSize + 1;
		if ( gridSize > 65535) {
                         
            printf("inputLength to large %i > 65535  ", gridSize);    
            return -1;
			}
		dim3 DimGrid(gridSize, 1, 1);
		dim3 DimBlock(blockSize, 1, 1);

    
  //  wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here

		vecAdd<<<DimGrid,DimBlock>>>( deviceInput1, deviceInput2, deviceOutput,  inputLength);
	
    cudaThreadSynchronize();
 //   wbTime_stop(Compute, "Performing CUDA computation");
    
 //   wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	
		wbCheck(cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost));

 //   wbTime_stop(Copy, "Copying output memory to the CPU");

 //   wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here

		cudaFree(deviceInput2); cudaFree(deviceInput1); cudaFree(deviceOutput);
	
//    wbTime_stop(GPU, "Freeing GPU Memory");

  //  wbSolution(args, hostOutput, inputLength);
		int ii;
	for ( ii = 0; ii < inputLength; ii++)
    {
		if ( hostCheckOutput[ii] != hostOutput[ii]) {
        printf("%d    %d  \n",hostOutput[ii],hostCheckOutput[ii]);
		}
		else
		{ 
			//printf(" Good  %i \n",ii);
		}
		
    }
	printf(" Done     %i \n",ii);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

