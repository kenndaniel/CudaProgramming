
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include	<wb.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>
#include "CL\opencl.h"


//@@ OpenCL Kernel
#define wbCheck(stmt) do {                                               	\
        cl_int err = stmt;                                               	\
        if (err != CL_SUCCESS) {                                            \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                     \
            wbLog(ERROR, "Got OpenCL error ...  ", get_error_string(err));  \
            return -1;                                                      \
        }                                                                   \
    } while(0)
		
const char * get_error_string(cl_int err){
         switch(err){
             case 0: return "CL_SUCCESS";
             case -1: return "CL_DEVICE_NOT_FOUND";
             case -2: return "CL_DEVICE_NOT_AVAILABLE";
             case -3: return "CL_COMPILER_NOT_AVAILABLE";
             case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
             case -5: return "CL_OUT_OF_RESOURCES";
             case -6: return "CL_OUT_OF_HOST_MEMORY";
             case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
             case -8: return "CL_MEM_COPY_OVERLAP";
             case -9: return "CL_IMAGE_FORMAT_MISMATCH";
             case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
             case -11: return "CL_BUILD_PROGRAM_FAILURE";
             case -12: return "CL_MAP_FAILURE";

             case -30: return "CL_INVALID_VALUE";
             case -31: return "CL_INVALID_DEVICE_TYPE";
             case -32: return "CL_INVALID_PLATFORM";
             case -33: return "CL_INVALID_DEVICE";
             case -34: return "CL_INVALID_CONTEXT";
             case -35: return "CL_INVALID_QUEUE_PROPERTIES";
             case -36: return "CL_INVALID_COMMAND_QUEUE";
             case -37: return "CL_INVALID_HOST_PTR";
             case -38: return "CL_INVALID_MEM_OBJECT";
             case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
             case -40: return "CL_INVALID_IMAGE_SIZE";
             case -41: return "CL_INVALID_SAMPLER";
             case -42: return "CL_INVALID_BINARY";
             case -43: return "CL_INVALID_BUILD_OPTIONS";
             case -44: return "CL_INVALID_PROGRAM";
             case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
             case -46: return "CL_INVALID_KERNEL_NAME";
             case -47: return "CL_INVALID_KERNEL_DEFINITION";
             case -48: return "CL_INVALID_KERNEL";
             case -49: return "CL_INVALID_ARG_INDEX";
             case -50: return "CL_INVALID_ARG_VALUE";
             case -51: return "CL_INVALID_ARG_SIZE";
             case -52: return "CL_INVALID_KERNEL_ARGS";
             case -53: return "CL_INVALID_WORK_DIMENSION";
             case -54: return "CL_INVALID_WORK_GROUP_SIZE";
             case -55: return "CL_INVALID_WORK_ITEM_SIZE";
             case -56: return "CL_INVALID_GLOBAL_OFFSET";
             case -57: return "CL_INVALID_EVENT_WAIT_LIST";
             case -58: return "CL_INVALID_EVENT";
             case -59: return "CL_INVALID_OPERATION";
             case -60: return "CL_INVALID_GL_OBJECT";
             case -61: return "CL_INVALID_BUFFER_SIZE";
             case -62: return "CL_INVALID_MIP_LEVEL";
             case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
             default: return "Unknown OpenCL error";
         }
     }


int main(int argc, char ** argv) {
//    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
	float * hostCheckOutput;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;
	const char* kernelSource = "__kernel void VecAdd(__global const float* a, __global const float* b, \n"
		"__global float* c,  int iNumElements){  int iglobal = get_global_id(0); \n"
	"if (iglobal >= iNumElements){ return;  }      c[iglobal] = a[iglobal]+  b[iglobal];}";
    
//	args = wbArg_read(argc, argv);

//    wbTime_start(Generic, "Importing data and creating memory on host");
//    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
//    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);

		hostCheckOutput = (float *) malloc(inputLength * sizeof(float));
		hostInput1 = (float *) malloc(inputLength * sizeof(float));
		hostInput2 = (float *) malloc(inputLength * sizeof(float));
			 for (int i = 0; i < inputLength; i++)
    {
        hostInput1[i] = 0;
        hostInput2[i] = i;
		hostCheckOutput[i] = hostInput1[i] + hostInput2[i];
       
    }

    hostOutput = (float *) malloc(inputLength * sizeof(float));

 //   wbTime_stop(Generic, "Importing data and creating memory on host");

//    wbLog(TRACE, "The input length is ", inputLength);
	

//	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	cl_uint numPlatforms;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_platform_id * platforms;
	platforms = (cl_platform_id *) malloc(inputLength * sizeof(cl_platform_id));
	clGetPlatformIDs(numPlatforms, platforms, NULL);
	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0};
//	

	cl_int clerr = CL_SUCCESS;
//	wbCheck(clerr);

	cl_context clctx = clCreateContextFromType(properties,CL_DEVICE_TYPE_GPU,NULL,NULL,&clerr);

	size_t pmsz;

	clerr = clGetContextInfo(clctx,CL_CONTEXT_DEVICES,0,NULL,&pmsz);

	cl_device_id* cldevs = (cl_device_id *) malloc(pmsz);
	
	clerr = clGetContextInfo(clctx,CL_CONTEXT_DEVICES, pmsz,cldevs, NULL);
	
	cl_command_queue clcmdq = clCreateCommandQueue(clctx,cldevs[0], 0, &clerr);
//	wbCheck(clerr);
	cl_program clpgm;
	clpgm = clCreateProgramWithSource(clctx, 1, &kernelSource,NULL, &clerr);


	char clcompileflags[4096];
	sprintf(clcompileflags, "-cl-mad-enable");
	clerr = clBuildProgram(clpgm, 0, NULL, clcompileflags,NULL, NULL);

	cl_kernel clkern = clCreateKernel(clpgm, "VecAdd", &clerr);

	
	cl_mem d_a;
	cl_mem d_b;
	cl_mem d_out;
	int size = inputLength * sizeof(float);
	// This will create a buffer in each device int the context
	// "| CL_MEM_COPY_HOST_PTR" causes an automatic copy
	d_a = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR ,size, hostInput1,&clerr);

	d_b = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, hostInput2,&clerr);

	d_out = clCreateBuffer(clctx,CL_MEM_READ_WRITE , size, hostOutput,&clerr);
	
	clerr = clSetKernelArg(clkern,0,sizeof(cl_mem),(void *)&d_a);
//		wbCheck(clerr);
	
	clerr = clSetKernelArg(clkern,1,sizeof(cl_mem),(void *)&d_b);
//		wbCheck(clerr);
	
	clerr = clSetKernelArg(clkern,2,sizeof(cl_mem),(void *)&d_out);
//		wbCheck(clerr);

	clerr= clSetKernelArg(clkern, 3, sizeof(cl_int),(void *)&inputLength);
	//	wbCheck(clerr);
		printf(" 8 %d \n",clerr);
	cl_event event=NULL;
	
	size_t GlobalSize[3];
	size_t LocalSize[3];
	int blockSize = 256;
	
	LocalSize[0] = blockSize;
	GlobalSize[0] =((inputLength-1)/blockSize + 1)*blockSize;
	
	
	LocalSize[1] = 1; LocalSize[2] = 1; GlobalSize[1] =1; GlobalSize[2] =1;
//	 wbTime_stop(GPU, "Allocating GPU memory.");


    //@@ Launch the GPU Kernel here
//		    wbTime_start(Compute, "Performing CUDA computation");
	clerr= clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL,GlobalSize, LocalSize, 0, NULL, &event);
//			wbCheck(clerr);

	clerr= clWaitForEvents(1, &event);
//			wbCheck(clerr);

//    wbTime_stop(Compute, "Performing CUDA computation");
    
 //   wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	clerr = clEnqueueReadBuffer(clcmdq, d_out, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
//			wbCheck(clerr);

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
//    wbTime_stop(Copy, "Copying output memory to the CPU");

//    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_out);

//    wbTime_stop(GPU, "Freeing GPU Memory");

//    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
