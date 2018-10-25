//
#include "KMeansHeader.h"


__global__ void pointsMovementCalKernel(int size, double* dev_initPointsCordinates,double* dev_pointsVelocityArr, double* dev_currentPointsCordinates, double time)
{
	int processId = threadIdx.x;
	dev_currentPointsCordinates[processId] = dev_initPointsCordinates[processId] + (dev_pointsVelocityArr[processId] * time);
}

boolean calPointsCordsCuda(double time, double* initPointsCordinates, double* pointsVelocityArr, double* currentPointsCordniates, int size)
{
	cudaError_t cudaStatus;
	int counter = 0;

	cudaStatus = computePointsCordinates(time,initPointsCordinates, pointsVelocityArr, currentPointsCordniates, size);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!"); fflush(stdout);
		return FALSE;
	}

	return TRUE;
}

void error(double* dev_currentPointsCordinates, double* dev_pointsVelocityArr, double* dev_initPointsCordinates)
{
	cudaFree(dev_currentPointsCordinates);
	cudaFree(dev_pointsVelocityArr);
	cudaFree(dev_initPointsCordinates);
}

cudaError_t computePointsCordinates(double time,double* initPointsCordinates , double* pointsVelocityArr, double* currentPointsCordniates, int size)
{
	cudaError_t cudaStatus;
	double* dev_currentPointsCordinates = 0;
	double* dev_pointsVelocityArr = 0;
	double* dev_initPointsCordinates = 0;
	int parts = size / 1000;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}
	
	// Allocate GPU buffers for Points vector    .
	cudaStatus = cudaMalloc((void**)&dev_currentPointsCordinates, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!"); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}

	cudaStatus = cudaMalloc((void**)&dev_pointsVelocityArr, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!"); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}

	cudaStatus = cudaMalloc((void**)&dev_initPointsCordinates, size * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!"); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}

	// Copy input vectors from host memory to GPU buffers.
	//cudaStatus = cudaMemcpy(dev_middleResultArr, middleResultArr, RANGE_SIZE * NUM_OF_THREADS * sizeof(int), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_currentPointsCordinates, currentPointsCordniates, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!"); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}

	cudaStatus = cudaMemcpy(dev_pointsVelocityArr, pointsVelocityArr, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!"); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}

	cudaStatus = cudaMemcpy(dev_initPointsCordinates, initPointsCordinates, size * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!"); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}

	
	pointsMovementCalKernel << <parts, size/parts >> >(size, dev_initPointsCordinates, dev_pointsVelocityArr, dev_currentPointsCordinates, time);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); fflush(stdout);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr, dev_initPointsCordinates);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		error(dev_currentPointsCordinates, dev_pointsVelocityArr , dev_initPointsCordinates);
	}
	return cudaStatus;
}


