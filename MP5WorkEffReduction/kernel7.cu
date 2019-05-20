
// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include	<wb.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>

#define wbCheck(stmt)  do { cudaError_t err = stmt; if (err != cudaSuccess) {printf( "XXX%&%& Failed to run stmt ", stmt); return -1;}} while(0)    

#define BLOCK_SIZE 512 //@@ You can change this ********

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


__global__ void scan(float * input, float * output,  int len) {
    //@@ Load a segment of the input vector into shared memory
			__shared__ float sh_input[2048];
						 

			int tx = threadIdx.x; 
			int tx2= tx + blockDim.x;
			int bdimx = blockDim.x;
			int i = 2*blockIdx.x*blockDim.x + tx;

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
				//output[Col1] = sh_input[tx]; output[Col2] = sh_input[tx2];

				unsigned int stride; int index;
		//  @@ Traverse the reduction tree down
			for (stride = 1;stride <= 2*bdimx ; stride *= 2)				   
				{
					index = (tx +1)* stride*2 -1;
					if (index  < 2*bdimx)
						sh_input[index] += sh_input[index-stride];
					__syncthreads();
				}

			//@@ Traverse the reduction tree up
				for ( stride = bdimx/2;  stride > 0; stride/=2)
				{
					__syncthreads();
					index = (tx +1)* stride*2 -1;
					if (index + stride < 2*bdimx)
						sh_input[index+stride] += sh_input[index];
				}

		//@@ Write the computed sum of the block to the output vector at the 
				//@@ correct index
					__syncthreads();
					output[i] = sh_input[tx];
					if ( i + bdimx < len)
					{	
						output[i + bdimx] = sh_input[tx2];
					}

 }
__global__ void vecAdd(float * in1, int offset, int len) { 
    //@@ Insert code to implement vector addition here
	int i =  threadIdx.x;
	
	if( (offset + i) <len ) in1[offset + i] = in1[offset + i]+in1[offset-1];
	if( (offset + i + blockDim.x ) <len ) in1[offset + i+ blockDim.x] = in1[offset + i+ blockDim.x]+in1[offset-1];
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
	float testInput[51200];

 //   args = wbArg_read(argc, argv);

//    wbTime_start(Generic, "Importing data and creating memory on host");
//    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);
	numInputElements = 4087;

	printf("\n Elements %d", numInputElements);
	for(int i = 0; i < numInputElements; ++i)
	{
		testInput[i] = 1;
	}
	
	hostInput = &testInput[0];

	numOutputElements = numInputElements;
	
	hostOutput = (float*) malloc(numOutputElements * sizeof(float));
	float testOutput = 0;

	//for (int j =0; j < numOutputElements; ++j)
	//{
	//	for (int i = 0; i < 2*BLOCK_SIZE; ++i)
	//	{
	//		testOutput += hostInput[i];
	//	} 
	//	
	//	printf("%3.0f ",testOutput);
	//} 
	//printf("\n");
		

 //   wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start( "Allocating GPU memory.");
    //@@ Allocate GPU memory here

	//wbCheck(cudaMalloc((void **) &deviceInput, numInputElements* sizeof(float)));  // Input

	//numOutputElements = numInputElements;

	//wbCheck(cudaMalloc((void **) &deviceOutput, numOutputElements* sizeof(float)));  // Output

 //   wbTime_stop( "Allocating GPU memory.");

 //   wbTime_start( "Copying input memory to the GPU.");
 //   //@@ Copy memory to the GPU here
	//wbCheck(cudaMemcpy(deviceInput, hostInput, numInputElements* sizeof(float), cudaMemcpyHostToDevice));
	int numElements=numInputElements;

	    wbTime_start( "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop( "Allocating GPU memory.");

    wbTime_start("Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop("Clearing output memory.");

    wbTime_start( "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop( "Copying input memory to the GPU.");

    wbTime_stop( "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here

		int j; int iBlockSize;
		for (j = 16; j <= 1024; j = j*2)
		{
			int k = numElements / j;
			int kk = numElements % j;
			if ((k == 0 && kk < j) || ( k == 1 && kk == 0) ) { break;}
			iBlockSize = j;
		}

		int gridSizex =((numElements)-1)/j + 1;
	    int gridSizey =1;

		if (  gridSizex > 65535 ) {

			printf("Grid too large %i x %i > 65535  \n", gridSizex,gridSizey);    
			return -1;
		}
	dim3 DimGrid(gridSizex, gridSizey);
	dim3 DimBlock(iBlockSize);

	//float * deviceBlockEnd;
	//float * hostBlockEnd;
	// hostBlockEnd = (float*) malloc(gridSizex * sizeof(float));
	//wbCheck(cudaMalloc((void **) &deviceBlockEnd, gridSizex* sizeof(float)));  // Output

    wbTime_start( "Performing CUDA computation");

    //@@ Launch the GPU Kernel here
	scan<<<DimGrid,DimBlock>>>( deviceInput, deviceOutput,  numElements);

	// Correct for block boundaries
	int limit;
	for (int iadd =1; iadd < gridSizex; ++iadd)
	{
		cudaDeviceSynchronize();

		if (( (iadd+1)* 2*iBlockSize) > numElements)
		{ limit = numElements;}
		else
		{ limit = (iadd+1)* 2*iBlockSize;}

		vecAdd<<<DimGrid,DimBlock>>>( deviceOutput, iadd*2*iBlockSize,  limit);
	}

	cudaDeviceSynchronize();

    wbTime_stop( "Performing CUDA computation");

    wbTime_start( "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

    wbTime_stop( "Copying output memory to the CPU");
	wbCheck(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));
	// wbCheck(cudaMemcpy(hostBlockEnd, deviceBlockEnd, gridSizex* sizeof(float), cudaMemcpyDeviceToHost));
    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    
			//for (ii = 0 ; ii < numElements; ii++) {

			//	printf("%3.0f ",hostOutput[ii]);
			//}

			//printf(" \n Block Ends \n");
			//for (ii = 0 ; ii < gridSizex; ii++) {

			//	printf("%3.0f ",hostBlockEnd[ii]);
			//}

			FILE * fp;
			fp = fopen ("file.txt", "w+");

			for (ii = 0 ; ii < numOutputElements; ii++) {
				fprintf(fp, "  >> %d  %3.0f \n",ii, hostOutput[ii]);
			}
			fclose(fp);
	
    wbTime_start( " \n Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceOutput); cudaFree(deviceInput); //cudaFree(deviceBlockEnd);

    wbTime_stop( "Freeing GPU Memory");

    //wbSolution( hostOutput, 1);

    free(hostOutput);

    return 0;
}
