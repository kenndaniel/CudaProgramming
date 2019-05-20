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

//		hostOutput = (float *)malloc(inputLength * sizeof(float));
	 // Using Pinned Memory
	   wbCheck( cudaHostAlloc((void **) &hostOutput, inputLength * sizeof(float),cudaHostAllocDefault));
		hostCheckOutput= (float *)malloc(inputLength * sizeof(float));
		printf("1  %i \n", inputLength);

 //   wbTime_stop(Generic, "Importing data and creating memory on host");
		 for (int i = 0; i < inputLength; i++)
    {
        hostInput1[i] = 0;
        hostInput2[i] = i;
		hostCheckOutput[i] = hostInput1[i] + hostInput2[i];
       
    }
		 printf("2 \n");
 //   wbLog(TRACE, "The input length is ", inputLength);
		 //  Async loading supported?
		 int dev_count;
		 cudaDeviceProp prop;
		 cudaGetDeviceCount( &dev_count);
		 for (int i=0; i < dev_count; i++)
		 {
			 cudaGetDeviceProperties(&prop,i);
			 if(prop.deviceOverlap) printf(" Async loading and Streams supported \n");
		 }


//	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
		 int numStreams = 4;

		 cudaStream_t stream0, stream[4];
		 wbCheck(cudaStreamCreate(&stream0));

		 for (int i = 0; i < numStreams; ++i)
		 {
			 wbCheck(cudaStreamCreate(&stream[i]));
		 }

		int size = inputLength * sizeof(float); 
	

		wbCheck(cudaMalloc((void **) &deviceInput1, size));  // Input
		wbCheck(cudaMalloc((void **) &deviceInput2, size));  // Input
	
		wbCheck(cudaMalloc((void **) &deviceOutput, size));  // Output
		printf("3 \n");
 //   wbTime_stop(GPU, "Allocating GPU memory.");

 //   wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
		int SegSize = 40960;
		size = SegSize* sizeof(float);
		int iStream;
		int iteration = inputLength/SegSize +1;
		for (int i=0; i < iteration ; i++) 
		{
			iStream = i % numStreams;
			//stream[iStream] = stream0;
			if (inputLength == SegSize*i) // inputLength multiple of SegSize
			{
				break;
			}
			else if(inputLength < SegSize*(i+1)) // process the small amount not yet processed
			{
				size = (inputLength - i*SegSize)* sizeof(float);
			}

		wbCheck(cudaMemcpyAsync(deviceInput1, hostInput1+i*SegSize, size, cudaMemcpyHostToDevice,stream[iStream]));
		wbCheck(cudaMemcpyAsync(deviceInput2, hostInput2+i*SegSize, size, cudaMemcpyHostToDevice,stream[iStream]));
		printf(" %d  %d %d \n", i, size, iStream);

    //@@ Initialize the grid and block dimensions here
		int blockSize = 1024;
		int gridSize =(SegSize-1)/blockSize + 1;
		if ( gridSize > 65535) {
                         
            printf("inputLength to large %i > 65535  ", gridSize);    
            return -1;
			}
		dim3 DimGrid(gridSize, 1, 1);
		dim3 DimBlock(blockSize, 1, 1);

    
 

		vecAdd<<<DimGrid,DimBlock,0,stream[iStream]>>>( deviceInput1, deviceInput2, deviceOutput,  size);
		
		cudaThreadSynchronize();
   

	
		wbCheck(cudaMemcpyAsync(hostOutput+i*SegSize, deviceOutput, size, cudaMemcpyDeviceToHost,stream[iStream]) );
		}


		cudaFree(deviceInput2); cudaFree(deviceInput1); cudaFree(deviceOutput);
	

		int ii;
		float sum = 0.;
		for ( ii = 0; ii < inputLength; ii++)
		{
			if ( hostCheckOutput[ii] != hostOutput[ii]) {
				printf("%d    %d  \n",hostOutput[ii],hostCheckOutput[ii]);

			}
			else
			{ 
				sum = sum + hostOutput[ii];
				// printf(" Good  %i \n",ii);
			}

		}
		printf(" Done     %d  %f \n",ii, sum);
		free(hostInput1);
		free(hostInput2);
		cudaFreeHost(hostOutput);

    return 0;
}

