

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include	<wb.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>

#define wbCheck(stmt)  do { cudaError_t err = stmt; if (err != cudaSuccess) {printf( "XXX%&%& Failed to run stmt ", stmt); return -1;}} while(0)    


#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#define wbLog(level, msg)   printf(msg)
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

#define TILE_WIDTH 14
#define Mask_width  5
#define Mask_radius 2
// Used to fake the input data
#define maskRows 5
#define maskColumns 5
#define imageChannels 3
 #define clamp(x,start,end)  (x > start ? x:start) < end ? (x > start ? x:start) : end 
    

//@@ INSERT CODE HERE
__global__ void imageConvolution(float *deviceInputImageData, float *deviceOutputImageData,
				const float * __restrict__ MaskData, 
				int imgWidth, int imgHeight, int imgChannels) 
{
			// Note that the shared memory is larger than the output tile to accomodate the skirt elements
			__shared__ float sharedImageData[TILE_WIDTH+2*(Mask_radius)][TILE_WIDTH+2*(Mask_radius)][3];
			
	
			//int Mask_radius = 2;
	
			int tx = threadIdx.x; 
			int ty = threadIdx.y;
		
			int by = blockIdx.y;
			int bx = blockIdx.x;
			// TILE_WIDTH is the output tile width Number of blocks * TILE_WIDTH = Image Width

			int imgRowO = by * (blockDim.y -2*Mask_radius) + ty ;
			int imgColO =  bx * (blockDim.x -2*Mask_radius) + tx ; 
			int imgRowI = imgRowO - Mask_radius;
			int imgColI = imgColO - Mask_radius;

			if(( imgRowI < imgHeight ) && (imgColI  < imgWidth) && imgColI  >= 0 && imgRowI  >= 0)
			{
			// Collaborative loading of input image block  
			// Three statements to eliminate "for loop" conditional test
			sharedImageData[ty][tx][0] = deviceInputImageData[(( imgRowI )* imgWidth + imgColI )* imgChannels + 0];	
			sharedImageData[ty][tx][1] = deviceInputImageData[(( imgRowI )* imgWidth + imgColI )* imgChannels + 1];
			sharedImageData[ty][tx][2] = deviceInputImageData[(( imgRowI )* imgWidth + imgColI )* imgChannels + 2];

			} 
			else
			{ // Ghost pixels - outside of image
			sharedImageData[ty][tx][0] = 0.0f;
			sharedImageData[ty][tx][1] = 0.0f;
			sharedImageData[ty][tx][2] = 0.0f;
			
			}
			
		   __syncthreads();

	// offset for halo and ghost elements
     int  tylRow= ty;
	 if( tylRow > 2 && tylRow < TILE_WIDTH-2 )
    {     
       int tylCol  = tx;
	   if ( tylCol > 2 && tylCol < TILE_WIDTH-2)
	   {

		  for (  int  ichan  = 0 ; ichan < imgChannels ; ++ichan)
		  {            
			 float accum = 0.0f;
             for (int  mRow  = -Mask_radius ; mRow <= Mask_radius ; mRow++ ) 
		     {   // Scan down a row of the mask

               for (  int  mCol  = -Mask_radius ; mCol <= Mask_radius ; mCol++ ) 
			   {   // Scan across a column of the mask

                 int xOffset = tylCol + mCol;
                 int yOffset = tylRow + mRow;

					  float maskValue = MaskData[(mRow+Mask_radius) *Mask_width + mCol+Mask_radius];
					 
				   // Scan through color channels
					  float imagePixel = sharedImageData[yOffset][xOffset][ichan];
					  accum += imagePixel * maskValue;
				}
				
            }
			deviceOutputImageData[(( imgRowO )* imgWidth + imgColO )* imgChannels + ichan] = clamp(accum, 0.0f, 1.0f) ;
          }
        }
	}
}

#define imageChannels 3
#define imageHeight 14
#define imageWidth 14
int address(int ih, int iw, int ic)
{
	return (imageWidth*imageChannels*ih + imageChannels*iw + ic);
}

int main(int argc, char* argv[]) {
    char* args;
    //int maskRows;
    //int maskColumns;
    //int imageChannels;
    //int imageWidth;
    //int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    float inputImage[imageHeight][imageWidth][imageChannels];
    float outputImage[imageHeight][imageWidth][imageChannels];
	float testoutputImage[imageHeight][imageWidth][imageChannels];
    float * hostInputImageData;
	float * TestOutputImageData = &testoutputImage[0][0][0];
    float * hostOutputImageData;
    // float * hostMaskData;
    float * deviceInputImageData;
	float * deviceMaskData;
    float * deviceOutputImageData;
   // float  deviceMaskData[25] = { 1,1,1,1,1, 1,1,2,1,1 ,1,2,2,2,1, 1,1,2,1,1, 1,1,1,1,1 };
	float * hostmask;
	float mask[25] = { 1,1,1,1,1, 1,1,2,1,1 ,1,2,2,2,1, 1,1,2,1,1, 1,1,1,1,1 };
	float  hostMaskData[25]= { 1,1,1,1,1, 1,1,2,1,1 ,1,2,2,2,1, 1,1,2,1,1, 1,1,1,1,1 };
	

    //args = wbArg_read(argc, argv); /* parse the input arguments */

    //inputImageFile = wbArg_getInputFile(args, 0);
    //inputMaskFile = wbArg_getInputFile(args, 1);

  //  inputImage = wbImport(inputImageFile);
	hostInputImageData = &inputImage[0][0][0];
	hostOutputImageData = &outputImage[0][0][0];
	//deviceInputImageData = hostInputImageData;
	//deviceOutputImageData = hostOutputImageData;

	for(int ih = 0; ih < imageHeight; ++ih)
	{int ic; int iw;
		for( iw = 0 ; iw < imageWidth; ++iw)
		{
			for( ic= 0; ic < imageChannels; ++ic)
			{
			hostInputImageData[ address(ih,iw,ic)] = 2.0f;
			//hostOutputImageData[address(ih,iw,ic)] = 0.0;
			}
			printf("%d#",address(ih,iw,ic));
		}
		printf("$%d$ \n",ih);
		
	}
	printf(" --- Host Input Image ----\n");
	int iCheck;
	for(int ih = 0; ih < imageHeight; ++ih)
	{
		for(int iw = 0 ; iw < imageWidth; ++iw)
		{
			for(int ic = 0; ic < imageChannels; ++ic)
			{
			printf("%1.0f",hostInputImageData[ address(ih,iw,ic)]);
			
			}
			printf("$");
		}
		printf("\n");
	}
	    //int maskRows;
    //int maskColumns;
    //int imageChannels;
    //int imageWidth;
    //int imageHeight;

	int maskWidth = Mask_width; int iCol; int ichan ; int mRow ; int mCol; int xOffset; int yOffset;
    int maskRadius = Mask_width/2; //# this is integer division, so the result is 2
    for ( int  iRow= 0 ; iRow<  imageHeight ; ++iRow)

	{
      for (  iCol  = 0 ; iCol < imageWidth ; ++iCol) 
	  {
		// Loop for one pixel for all channels
        for (    ichan  = 0 ; ichan < imageChannels ; ++ichan)
		{ // Scan through color channels

          float accum = 0;
          for (  mRow  = -maskRadius ; mRow <= maskRadius ; mRow++ ) 
		  {   // Scan down a column

            for (    mCol  = -maskRadius ; mCol <= maskRadius ; mCol++ ) 
			{   // Scan across a column

              xOffset = iCol + mCol;
              yOffset = iRow+ mRow;
              if ( xOffset >= 0 && xOffset < imageWidth &&  yOffset >= 0 && yOffset < imageHeight )
			  { // inside the image - check for ghost elemets
				 // if () 
				  { // Check for halo elements
				   float imagePixel = hostInputImageData[(yOffset * imageWidth + xOffset)  * imageChannels + ichan];
	               float maskValue = hostMaskData[(mRow+maskRadius) *maskWidth + mCol+maskRadius];
					// printf("%1.0f",maskValue);
		            accum += imagePixel * maskValue;
				  }
              }
            }//printf("\n");
          }
          //# pixels are in the range of 0 to 1
          TestOutputImageData[(iRow* imageWidth + iCol) *imageChannels + ichan] = accum; //clamp(accum, 0, 1) ;
        }
      }
	}
				printf("\n");

	for(int ih = 0; ih < imageHeight; ++ih)
	{
		for(int iw = 0 ; iw < imageWidth; ++iw)
		{
			for(int ic = 0; ic < imageChannels; ++ic)
			{
			printf("%1.0f",TestOutputImageData[ address(ih,iw,ic)]);
			
			}
			printf("$");
		}
		printf("\n");
	}
//    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    //assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    //assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    //imageWidth = wbImage_getWidth(inputImage);
    //imageHeight = wbImage_getHeight(inputImage);
    //imageChannels = wbImage_getChannels(inputImage);

   // outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

 //   hostInputImageData = wbImage_getData(inputImage);
 //   hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start("Doing GPU Computation (memory + compute) \n");

    wbTime_start("Doing GPU memory allocation \n");
	int size =  imageWidth * imageHeight * imageChannels * sizeof(float);
    wbCheck(cudaMalloc((void **) &deviceInputImageData,size));
    wbCheck(cudaMalloc((void **) &deviceOutputImageData,size));
    wbCheck(cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float)));
    wbTime_stop("Doing GPU memory allocation");


    wbTime_start("Copying data to the GPU \n");
	printf(" Input image ");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
	printf(" Mask \n");
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);

	///////////////////// For Test Only
	/* cudaMemcpy(deviceOutputImageData,
               hostOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);*/
	 /////////////////// For Testing Only
    wbTime_stop("Copying data to the GPU \n");


    wbTime_start("Doing the computation on the GPU \n");
    //@@ INSERT CODE HERE
	// Initialize the grid and block dimensions here
	// TILE_WIDTH is the output tile width Number of blocks * TILE_WIDTH = Image Width

  	int gridSizex =((imageWidth)-1)/TILE_WIDTH + 1;
	int gridSizey =((imageHeight)-1)/TILE_WIDTH + 1;

		if ( gridSizex > 65535 || gridSizey > 65535 ) {

			printf("Grid too large %i x %i > 65535  \n", gridSizex,gridSizey);    
			return -1;
		}
	dim3 DimGrid(gridSizex, gridSizey,1);
	dim3 DimBlock(TILE_WIDTH+2*Mask_radius, TILE_WIDTH+2*Mask_radius, 1);

    wbTime_start( "Performing CUDA computation \n");
    //@@ Launch the GPU Kernel here
	 imageConvolution<<<DimGrid,DimBlock>>>( deviceInputImageData, deviceOutputImageData,deviceMaskData, 
				imageWidth, imageHeight, imageChannels);
	
    cudaThreadSynchronize();
    wbTime_stop( "Performing CUDA computation \n");
    wbTime_stop("Doing the computation on the GPU \n");


    wbTime_start("Copying data from the GPU \n");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop("Copying Answer from the GPU \n");

for(int ih = 0; ih < imageHeight; ++ih)
	{
		for(int iw = 0 ; iw < imageWidth; ++iw)
		{
			for(int ic = 0; ic < imageChannels; ++ic)
			{
			printf("%1.0f",hostOutputImageData[ address(ih,iw,ic)]);
			
			}
			printf("$");
		}
	printf("\n");
}
    wbTime_stop("Doing GPU Computation (memory + compute) \n");

   // wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

   // free(hostMaskData);
    //wbImage_delete(outputImage);
    //wbImage_delete(inputImage);

    return 0;
}