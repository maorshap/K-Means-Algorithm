#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _CRT_SECURE_NO_WARNINGS
#define MASTER 0
#define TRANSFER_TAG 0
#define MID_TERMINATION_TAG 1
#define FINAL_TERMINATION_TAG 2
#define POINT_STRUCT_SIZE 6
#define CLUSTER_STRUCT_SIZE 3

typedef enum boolean { FALSE, TRUE };

typedef struct Cluster
{
	double x;
	double y;
	double diameter;//The largest distance between 2 points in the current cluster
};

typedef struct Point
{
	double x;
	double y;
	double vx;
	double vy;
	int currentClusterIndex;
	int previousClusterIndex;
};

void creationPointMPIType(MPI_Datatype* PointMPIType);
void creationClusterMPIType(MPI_Datatype* ClusterMPIType);
Point* readDataFromFile(int* totalNumOfPoints, int* K, int* limit, double* QM, double* T, double* dt);
void writeToFile(double t, double q, Cluster* clusters, int K);
void checkAllocation(void* pointer);
Cluster* initClusters(const Point* points, int K);
int getClosestClusterIndex(double x, double y, Cluster* clusters, int K);
void groupPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int numOfPoints, Cluster* clusters, int K);
double distancePointToPoint(double x1, double y1, double x2, double y2);
void calClusterCenter(Cluster* cluster, Point* clusterPoints, int clusterPointsSize);
double evaluateQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize);
double calClusterDiameter(Point* clusterPoints, int clusterPointsSize);
//void calPointsCordinates(Point* points, int totalNumOfPoints, double t);
void calPointsCordinates(Point* points, int N, double t);
double kMeansWithIntervals(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	int limit, double QM, double T, double dt, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr,
	int numprocs, double* time, double* currentPointsCordinates, double* initialPointsCordinates, double* pointsVelocityArr);
double kMeansAlgorithmMaster(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	int limit, int numOfProcs, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr, int numprocs);
void kMeansAlgorithmSlave(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K 
	, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr);
void gatherAllPoints(Point** pointsMat, int* clustersSize, int* totalClustersSize, int K, int numOfProcs, MPI_Datatype PointMPIType);
void returnPoints(int K, int* clustersSize, Point** pointsMat, MPI_Datatype PointMPIType);
void slaveOperation(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr, double* currentPointsCordinates ,double * initialPointsCordinates,double* pointsVelocityArr);
void initPointsInfoArrs(double* currentPointsCordinates, double* initialPointsCordinates,double* pointsVelocityArr, Point* points, int numOfPoints);
void refreshPointsCordinates(Point* points, int numOfPoints, double* currentPointsCordinates);



//Cuda functions
boolean calPointsCordsCuda(double time,  double* nitPointsCordinates, double* pointsVelocityArr, double* pointsCordniates, int size);
//boolean checkTerminationConditionCuda(const Point* points, int numOfPoints, MPI_Datatype PointMPIType);
cudaError_t computePointsCordinates(double time,double* initPointsCordinates,double* pointsVelocityArr, double* currentPointsCordniates, int size);
__global__ void pointsMovementCalKernel(int size, double* dev_initPointsCordinates , double* dev_pointsVelocityArr, double* dev_currentPointsCordinates, double time);
void error(double* dev_currentPointsCordinates, double* dev_pointsVelocityArr, double* dev_initPointsCordinates);
