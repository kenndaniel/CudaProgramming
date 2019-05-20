// OpenACC.cpp : main project file.

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>
using namespace System;

int main(array<System::String ^> ^args)
{
    Console::WriteLine(L"Hello World");

  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

 // wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)malloc(inputLength * sizeof(float));
  hostInput2 = (float *)malloc(inputLength * sizeof(float));
  hostOutput = (float *)malloc(inputLength * sizeof(float));
	for (int i=0; i< inputLength;i++) 
	{
		hostInput1[i] = i;
		hostInput2[i] = i;
	}
//#pragma acc parallel loop copyin(hostInput1[0:inputLength]) copyin(hostInput2[0:inputLength]) copyout(hostOutput[0:inputLength]) num_gangs(1024) num_workers(32)
#pragma acc kernels
	{
	#pragma acc loop
		for (int i=0; i< inputLength ;i++) 
		{
			hostOutput[i] = hostInput1[i]+hostInput2[i];
		}
	}
	 Console::WriteLine(L"Bye bye");
    return 0;
}
