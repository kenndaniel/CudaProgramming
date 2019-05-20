
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include	<wb.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>

#define wbCheck(stmt)  do { cudaError_t err = stmt; if (err != cudaSuccess) {printf( "Failed to run stmt ", stmt); return -1;}} while(0)    

#define wbLog(level, msg)   printf(msg)
clock_t start;
double diff;

#define Arow 128
#define Acol 64
#define Brow 64
#define Bcol 128


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

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
					   int iRow = blockIdx.y*blockDim.y+threadIdx.y;
					   int iCol = blockIdx.x*blockDim.x+threadIdx.x;
					   if(( iRow < numARows) && (iCol < numBColumns)) {
						   float Cvalue = 0.0;
						   for (int i = 0;i< numAColumns;++i)
						   {
							   Cvalue += A[iRow*numAColumns+i]*B[iCol+i*numBColumns];
						   }
						   C[iRow*numBColumns+iCol] = Cvalue;
					   }
}

int main(int argc, char ** argv) {
    //wbArg_t args;
    //float * hostA; // The A matrix
    //float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows = Arow; // number of rows in the matrix A
    int numAColumns = Acol; // number of columns in the matrix A
    int numBRows = Brow ; // number of rows in the matrix B
    int numBColumns = Bcol; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)
	float hostA[Arow][Acol];
	float hostB[Brow][Bcol];
	float testhostC[Arow][Bcol];
    //args = wbArg_read(argc, argv);

    wbTime_start("Importing data and creating memory on host");
   // hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
   // hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
	 //hostA = (float *)malloc( numARows*numAColumns * sizeof(float));
	 //hostB = (float *)malloc( numBRows*numBColumns * sizeof(float));
	 //hostC = (float *)malloc( numARows*numBColumns * sizeof(float));
    numCRows = 0;
    numCColumns = 0;
	printf("\n A Matrix \n");
		for (int i = 0; i < numARows; i++){
			for (int ii = 0; ii < numAColumns; ii++)
			{
				hostA[i][ii] = rand() / (float)RAND_MAX;
				printf(" %f ",hostA[i][ii] );
			}
			printf("\n");
		}

			printf("\n B Matrix \n");
		for (int i = 0; i < numBRows; i++){
			for (int ii = 0; ii < numBColumns; ii++)
			{
				hostB[i][ii] = rand() / (2*(float)RAND_MAX);
				printf(" %f ",hostB[i][ii] );
			}
			printf("\n");
		}
		float Cvalue,A,B;
		printf("\n Correct Answer \n");
		for (int iRow = 0; iRow < numARows; iRow++)
		{
			for (int iColi = 0; iColi < numBColumns; iColi++)
			{	
						Cvalue = 0.0;
						for (int i = 0; i < numAColumns; ++i)
						{
						/* A[Row, i] and B[i, Col] */
						A = hostA[iRow][i] ; B = hostB[i][iColi];
						Cvalue = hostA[iRow][i] * hostB[i][iColi] + Cvalue;
						//printf("%d %d %d A %f   B %f CV %f \n",iRow,i,iColi,A,B,Cvalue);
						}
					testhostC[iRow][iColi] = Cvalue;
					//printf(" %d %d ",iRow,iColi);
					printf(" %f ",testhostC[iRow][iColi] );
                }
			printf("\n");
		}
			printf("\n");


		
    //@@ Set numCRows and numCColumns

			int deviceCount;
			wbCheck(cudaGetDeviceCount(&deviceCount));
			for (int dev = 0; dev < deviceCount; dev++) 
			{
				cudaDeviceProp deviceProp;

				cudaGetDeviceProperties(&deviceProp, dev);

				if (dev == 0) {
					if (deviceProp.major == 9999 && deviceProp.minor == 9999) 
					{
						printf( "No CUDA GPU has been detected \n");
						return -1;
					} else if (deviceCount == 1) {
						printf( "There is 1 device supporting CUDA \n");
					} else {
						printf( "There are %d devices supporting CUDA",deviceCount);
					}

				}
			}
				numCRows = numARows;
				numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    wbTime_stop("Importing data and creating memory on host");
	if ( numAColumns != numBRows) {
                         
            printf("numAColumns must equal numBRows  %d != %d ", numAColumns,numBRows );    
            return -1;
			}

		hostC = (float *)malloc( numARows*numBColumns * sizeof(float));

    printf( "The dimensions of A are %d x %d \n" , numARows, numAColumns);
    printf( "The dimensions of B are %d x %d \n", numBRows, numBColumns);
    printf( "The dimensions of C are %d x %d \n", numCRows, numCColumns);

    wbTime_start("Allocating GPU memory.\n");
    //@@ Allocate GPU memory here
	int sizeA = numARows*numAColumns*sizeof(float);
	int sizeB = numBRows*numBColumns*sizeof(float);
	int sizeC = numARows*numBColumns*sizeof(float);

	wbCheck(cudaMalloc((void **) &deviceA, sizeA ));  // Input
	wbCheck(cudaMalloc((void **) &deviceB, sizeB));  // Input

	wbCheck(cudaMalloc((void **) &deviceC, sizeC));  // Output

    wbTime_stop( "Allocating GPU memory.\n");

    wbTime_start( "Copying input memory to the GPU.\n");
    //@@ Copy memory to the GPU here

		wbCheck(cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice));
		wbCheck(cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice));

    wbTime_stop( "Copying input memory to the GPU.\n");
	//hostC = &testhostC[0][0];
	//			for (int iRow = 0; iRow < numARows; iRow++)
	//	{
	//		for (int iColi = 0; iColi < numBColumns; iColi++)
	//		{
	//			printf(" %f(%f) ",hostC[iRow*numBColumns+iColi],testhostC[iRow][iColi]);
	//			//printf(" %f ",testhostC[iRow][iColi]);
	//		}
	//		printf(" \n");
	//}
    
    //@@ Initialize the grid and block dimensions here

	//int blockSize = 16*16;
	int gridSizex =((numBColumns)-1)/16 + 2;
	int gridSizey =((numARows)-1)/16 + 2;

		if ( gridSizex > 65535 || gridSizey > 65535 ) {

			printf("Grid too large %i  %i > 65535  \n", gridSizex,gridSizey);    
			return -1;
		}
	dim3 DimGrid(gridSizex, gridSizey, 1);
	dim3 DimBlock(16, 16, 1);

    wbTime_start( "Performing CUDA computation \n");
    //@@ Launch the GPU Kernel here
	 matrixMultiply<<<DimGrid,DimBlock>>>( deviceA,  deviceB,  deviceC,
			        numARows,  numAColumns,
			        numBRows,  numBColumns,
			        numCRows,  numCColumns);
	
    cudaThreadSynchronize();
    wbTime_stop( "Performing CUDA computation \n");
    
    wbTime_start( "Copying output memory to the CPU \n");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost));
    wbTime_stop( "Copying output memory to the CPU \n");
		for (int iRow = 0; iRow < numARows; iRow++)
		{
			for (int iColi = 0; iColi < numBColumns; iColi++)
			{
				printf(" %f ",hostC[iRow*numBColumns+iColi]);
			}
			printf(" \n");
		}
    wbTime_start( "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);
    wbTime_stop( "Freeing GPU Memory \n");

    //wbSolution(args, hostC, numCRows, numCColumns);
	printf(" Freeing host memory");
    //free(hostA);
    //free(hostB);
    //free(hostC);

    return 0;
}
