#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include	<wb.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>

#define wbCheck(stmt)  do { cudaError_t err = stmt; if (err != cudaSuccess) {printf( "XXX%&%& Failed to run stmt ", stmt); return -1;}} while(0)    
#define BLOCK_SIZE 512 //@@ You can change this

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#define wbLog(level, msg)   printf(" %s \n",msg)
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


__global__ void total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
						   __shared__ float sh_input[2*BLOCK_SIZE];
						   __shared__ float sh_output[2*BLOCK_SIZE];

					   int gx = gridDim.x;
					   int tx = threadIdx.x; 
					   int tx2= tx + blockDim.x;
					   int bdimx = blockDim.x;

						int start = 2*blockIdx.x*blockDim.x;
						int Col1 = start + tx;
						int Col2 = start + bdimx + tx;

						   if( Col2 < len)
						   {
								// Collaborative loading of A 
								sh_input[tx] = input[ Col1];
								sh_input[tx2] = input[ Col2];
						   } 
						   else if ( Col1 < len)
						   {	// Control divergence at the edge
							   sh_input[tx] = input[ Col1];
							   sh_input[tx2]= 0.0f;
						   }
						      else 
						   {	// Control divergence at the edge
							   sh_input[tx] = 0.0f;
							   sh_input[tx2]= 0.0f;
						   }
						    __syncthreads();
							//output[Col1] = sh_input[tx];

				//@@ Traverse the reduction tree
						   for (unsigned int stride = bdimx;
							   stride > 0; stride/=2)
						   {
							   __syncthreads();
							   if (tx < stride)
								   sh_input[tx] += sh_input[tx+stride];
						   }

				 //@@ Write the computed sum of the block to the output vector at the 
						    //@@ correct index
						//   sh_output[t] = sh_input[0];
						    output[blockIdx.x] = sh_input[0];
}

int main(int argc, char ** argv) {
    int ii;
//    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list
	float testInput[5120];
 //   args = wbArg_read(argc, argv);

//    wbTime_start(Generic, "Importing data and creating memory on host");
//    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);
	numInputElements = 2091;
	for(int i = 0; i < numInputElements; ++i)
	{
		testInput[i] = 1;
	}
	
	hostInput = &testInput[0];

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);

    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
	
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));
	float testOutput = 0;

	for (int j =0; j < numOutputElements; ++j)
	{
		for (int i = 0; i < 2*BLOCK_SIZE; ++i)
		{
			testOutput += hostInput[i];
		} 
		
		printf("%3.0f ",testOutput);
	} 
	printf("\n");
		

 //   wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start( "Allocating GPU memory.");
    //@@ Allocate GPU memory here

	wbCheck(cudaMalloc((void **) &deviceInput, numInputElements* sizeof(float)));  // Input
	// for testing purposes
	//numOutputElements = numInputElements;

	wbCheck(cudaMalloc((void **) &deviceOutput, numOutputElements* sizeof(float)));  // Output

    wbTime_stop( "Allocating GPU memory.");

    wbTime_start( "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceInput, hostInput, numInputElements* sizeof(float), cudaMemcpyHostToDevice));

    wbTime_stop( "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here

	  	int gridSizex =numOutputElements;
	    int gridSizey =1;

		if (  gridSizey > 65535 ) {

			printf("Grid too large %i x %i > 65535  \n", gridSizex,gridSizey);    
			return -1;
		}
	dim3 DimGrid(gridSizex, gridSizey);
	dim3 DimBlock(BLOCK_SIZE);

    wbTime_start( "Performing CUDA computation");

    //@@ Launch the GPU Kernel here
	total<<<DimGrid,DimBlock>>>( deviceInput, deviceOutput,  numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop( "Performing CUDA computation");

    wbTime_start( "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

    wbTime_stop( "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numOutputElements* sizeof(float), cudaMemcpyDeviceToHost));
    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    float accum = 0;
	for (ii = 0; ii < numOutputElements; ii++) {
        accum = hostOutput[ii] + accum;
		printf("%3.0f & %3.0f - ",hostOutput[ii], accum  );
    }
	printf("\n total = %3.0f - ",accum);
    wbTime_start( "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceOutput); cudaFree(deviceInput);

    wbTime_stop( "Freeing GPU Memory");

    //wbSolution( hostOutput, 1);

    free(hostOutput);

    return 0;
}
