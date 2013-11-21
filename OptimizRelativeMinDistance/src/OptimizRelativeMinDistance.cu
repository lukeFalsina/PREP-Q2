/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/** Thrust Libraries (not necessary..)
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
 */

#include "computation.cuh"

#define DEFAULT_SENSIBILITY 1000
#define THREADS_PER_BLOCK 512
/**
 * Default value used to target the results of all of those threads
 * whose delta results to be greater than 1. This trick helps to avoid
 * them to be considered as suitable results during the following
 * minimization step.
 */
#define INFINITY_VALUE 1000

/**
 * It is assumed that delta, which represents the relative minimum
 * distance of the random matrix M could only take values between
 * 0 and 1.
 */
#define MAX_DELTA_VALUE 1

/**
 * Help parameter used to enable/disable the detailed printing
 * of the couples (delta, n)
 */
bool verboseMode = false;

bool debugMode = false;

/**
 * General function used to find out the properties of the running CUDA device
 */
void DisplayHeaderDevice()
{
    const int Kb = 1024;
    const int Mb = Kb * Kb;

    int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for (int device = 0; device < deviceCount; ++device) {
    	cudaDeviceProp deviceProp;
    	cudaGetDeviceProperties(&deviceProp, device);
    	printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
       	printf("Global Memory:		%f Mb\n", (float) deviceProp.totalGlobalMem / Mb);
       	printf("Shared Memory:		%f Kb\n", (float) deviceProp.sharedMemPerBlock / Kb);
       	printf("Constant Memory:		%f Kb\n", (float) deviceProp.totalConstMem / Kb);
       	printf("Block registers:	%d\n", deviceProp.regsPerBlock);
       	printf("Warp Size:		%d\n", deviceProp.warpSize);
       	printf("Thread per block:	%d\n", deviceProp.maxThreadsPerBlock);
       	printf("Max block dimensions:	[ %d, %d, %d ]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       	printf("Max grid dimensions:	[ %d, %d, %d ]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

	}

}

int main(int argc, char **argv)
{

	// Parameter used to set the number of intervals used for the computation
	// of the optimal parameter
	int  sensibility;

	if (argc >= 2) sensibility = atoi(argv[1]);
	else sensibility = DEFAULT_SENSIBILITY;

	// Parameter used to set the maximum possible value for delta
	float maxDeltaThreshold;

	if (argc >= 3) maxDeltaThreshold = atof(argv[2]);
		else maxDeltaThreshold = MAX_DELTA_VALUE;

	if (argc == 4) verboseMode = true;

	// Show the properties of the CUDA devices..
	if(debugMode) DisplayHeaderDevice();

	size_t vectorSize = sensibility * sizeof(float);

	// Compute launching parameters for the kernel
	dim3 dimGrid((unsigned int) ceilf((float) sensibility/THREADS_PER_BLOCK ), 1, 1);
	dim3 dimBlock(THREADS_PER_BLOCK,1,1);

	// Allocating a raw vector to be processed on the device.
	float *rawTestAmountRowsVector;
	cudaMalloc(&rawTestAmountRowsVector, vectorSize);

	// Allocate the arbitrary infinite value in constant memory
	int infinityValue = INFINITY_VALUE;
	cudaMemcpyToSymbol(infinity, &infinityValue, sizeof(int));

	// Invoke the kernel for the computation of the amounted rows
	computeAmountRows<<<dimGrid, dimBlock>>>(rawTestAmountRowsVector, sensibility, maxDeltaThreshold);

	cudaThreadSynchronize();

	////////////////////////////////////////////////////////////////////////////////////////////////////
	// The minimum value is now extracted and all the correlated couples (delta, n) are printed out.
	////////////////////////////////////////////////////////////////////////////////////////////////////

	printf("\nEvaluation of the parameter Delta in order to find out the optimal amount of rows, N\n\n");
	printf("Test Condition:\nDomain of the parameter delta = [0, %f]\n", maxDeltaThreshold);
	printf("Number of samples considered = %d\nInterval between two adjacent delta samples = %f\n\n", sensibility, (float) 1/sensibility);

	float *testAmountRowsVector;
	testAmountRowsVector = (float *) malloc(vectorSize);

	cudaMemcpy(testAmountRowsVector, rawTestAmountRowsVector, vectorSize, cudaMemcpyDeviceToHost);

	float minComparison = infinityValue;
	float delta;
	int minIndex;

	for(int i=0; i< sensibility; i++) {
		if(testAmountRowsVector[i] < minComparison) {
			minIndex = i;
			minComparison = testAmountRowsVector[i];
		}

		delta = (float) i/sensibility + (float) 1/sensibility;

		if (delta <= maxDeltaThreshold && verboseMode) {
			printf("delta = %f -> n = %f \n", delta, testAmountRowsVector[i]);
		}
	}

	printf("\nThe minimum amount of rows is N = %f k obtained by taking delta = %f\n", minComparison, (float) minIndex/sensibility);

	if(argc < 2)
		printf("\nHELP\nYou can run the analysis with test values different from the default ones.\n"
			"In particular 3 parameters could be provided when launching the program:\n\t"
			"1. Numbers of intervals tested for Delta [positive integer value] (Default = 1000)\n\t"
			"2. Maximum value tested for Delta [float value between 0 and 1] (Default = 1)\n\t"
			"3. Enable/Disable Verbose mode [provide a non-null string to enable this mode.. e.g. v] (Default = Disable)\n");

	////////////////////////////////////////////////////////////////////////////////////////////////////

	// Wrap raw pointer with a device pointer
	//thrust::device_ptr<float> devTestAmountRowsVector(rawTestAmountRowsVector);

	// Instruction used to locate the position of the smallest amount of rows
	//ptr_to_smallest_value = thrust::min_element(devTestAmountRowsVector, devTestAmountRowsVector + sensibility);

	// Reduce operations to find the smallest amount of rows
	//float minAmountRows = thrust::reduce(devTestAmountRowsVector, devTestAmountRowsVector + sensibility, infinityValue, thrust::minimum<float>());

	//printf("The minimum amount of rows is N = %f k \n", minAmountRows);

	cudaFree(rawTestAmountRowsVector);

	return 0;
}
