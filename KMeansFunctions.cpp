
#include "KMeansHeader.h"
#include <mpi.h>

void writeToFile(double t, double q, Cluster* clusters, int K)
{
	const char* fileName = "C:\\Users\\mpi\\Documents\\Visual Studio 2015\\Projects\\finalProject\\finalProject\\output.txt";
	FILE* file;

	fopen_s(&file, fileName, "w");
	if (file == NULL) {
		printf("Couldnt open the file\n");
		exit(1);
	}

	fprintf_s(file, "First occurrence at t = %lf with q = %lf\nCenters of the clusters:\n", t, q);
	for (int i = 0; i < K; i++)
	{
		fprintf_s(file, "( %lf , %lf )\n", clusters[i].x, clusters[i].y);
	}

	fclose(file);

}

void creationPointMPIType(MPI_Datatype* PointMPIType)
{

	Point point;
	MPI_Datatype PointType[POINT_STRUCT_SIZE] = { MPI_DOUBLE, MPI_DOUBLE ,MPI_DOUBLE ,MPI_DOUBLE ,MPI_INT ,MPI_INT };
	int pointBlocklen[POINT_STRUCT_SIZE] = { 1, 1, 1, 1, 1, 1 };
	MPI_Aint pointDisp[POINT_STRUCT_SIZE];
	pointDisp[0] = (char *)&point.x - (char *)&point;
	pointDisp[1] = (char *)&point.y - (char *)&point;
	pointDisp[2] = (char *)&point.vx - (char *)&point;
	pointDisp[3] = (char *)&point.vy - (char *)&point;
	pointDisp[4] = (char *)&point.currentClusterIndex - (char *)&point;
	pointDisp[5] = (char *)&point.previousClusterIndex - (char *)&point;
	MPI_Type_create_struct(POINT_STRUCT_SIZE, pointBlocklen, pointDisp, PointType,PointMPIType);
	MPI_Type_commit(PointMPIType);
}

void creationClusterMPIType(MPI_Datatype* ClusterMPIType)
{
	Cluster cluster;
	MPI_Datatype ClusterType[CLUSTER_STRUCT_SIZE] = { MPI_DOUBLE, MPI_DOUBLE , MPI_DOUBLE };
	int clusterBlocklen[CLUSTER_STRUCT_SIZE] = { 1, 1, 1 };
	MPI_Aint clusterDisp[CLUSTER_STRUCT_SIZE];
	clusterDisp[0] = (char *)&cluster.x - (char *)&cluster;
	clusterDisp[1] = (char *)&cluster.y - (char *)&cluster;
	clusterDisp[2] = (char *)&cluster.diameter - (char *)&cluster;
	MPI_Type_create_struct(CLUSTER_STRUCT_SIZE, clusterBlocklen, clusterDisp, ClusterType, ClusterMPIType);
	MPI_Type_commit(ClusterMPIType);
}

void calClusterCordSum(Cluster* cluster, Point* clusterPoints, int clusterPointsSize, double* sumX, double* sumY)//Step 3 in K-Means algorithm
{
	int i;
	*sumX = 0;
	*sumY = 0;

	//Compute all the cluster points cordinates sums
	for (int i = 0; i < clusterPointsSize; i++)
	{
		*sumX += clusterPoints[i].x;
		*sumY += clusterPoints[i].y;
	}
}

Point* readDataFromFile(int* totalNumOfPoints, int* K, int* limit, double* QM, double* T, double* dt)
{
	int i;
	const char* POINTS_FILE = "C:\\Users\\mpi\\Documents\\Visual Studio 2015\\Projects\\finalProject\\finalProject\\test.txt";
	FILE* file = fopen(POINTS_FILE, "r");

	//Check if the file exist
	if (!file)
	{
		printf("could not open the file "); fflush(stdout);
		MPI_Finalize();
		exit(1);
	}

	fscanf(file, "%d %d %lf %lf %d %lf\n", totalNumOfPoints, K, T, dt, limit, QM);//Getting the supplied data from input file

	Point* points = (Point*)malloc(*totalNumOfPoints * sizeof(Point));
	checkAllocation(points);

	//Initalize points from file
	for (i = 0; i < *totalNumOfPoints; i++)
	{
		fscanf(file, "%lf %lf %lf %lf\n", &(points[i].x), &(points[i].y), &(points[i].vx), &(points[i].vy));
		points[i].currentClusterIndex = 0;
		points[i].previousClusterIndex = -1;
	}

	fclose(file);
	return points;
}

void checkAllocation(void* pointer)
{
	if (!pointer)
	{
		printf("Dynamic allocation failed\n"); fflush(stdout);
		MPI_Finalize();
		exit(1);
	}
}

Cluster* initClusters(const Point* points, int K)//Step 1 in K-Means algorithem
{
	int i;
	Cluster* clusters = (Cluster*)malloc(K * sizeof(Cluster));
	checkAllocation(clusters);

#pragma omp parallel for shared(clusters) 
	for (i = 0; i < K; i++)
	{
		clusters[i].x = points[i].x;
		clusters[i].y = points[i].y;
		clusters[i].diameter = 0;
	}

	return clusters;
}

int getClosestClusterIndex(double x, double y, Cluster* clusters, int K)//part of step 2 in K-Means algorithm
{
	int i, index = 0;
	double minDistance, tempDistance;

	minDistance = distancePointToPoint(x, y, clusters[0].x, clusters[0].y);

	for (i = 1; i < K; i++)
	{
		tempDistance = distancePointToPoint(x, y, clusters[i].x, clusters[i].y);
		if (tempDistance < minDistance)
		{
			minDistance = tempDistance;
			index = i;
		}
	}

	return index;
}

void groupPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int numOfPoints, Cluster* clusters, int K)//Step 2 in K-Means algorithm
{
	int i, tid;


		//Reset ClustersSize Array Cells
#pragma omp parallel for shared(clustersSize)
		for (i = 0; i < K; i++)
		{
			clustersSize[i] = 0;
		}

		//finding for each point his closet cluster
#pragma omp parallel for shared(points,clustersSize, pointsMat) private(tid)
		for (i = 0; i < numOfPoints; i++)
		{
			
			points[i].previousClusterIndex = points[i].currentClusterIndex;
			points[i].currentClusterIndex = getClosestClusterIndex(points[i].x, points[i].y, clusters, K);
		}
		for (i = 0; i < numOfPoints; i++)
		{
			clustersSize[points[i].currentClusterIndex]++;
			pointsMat[points[i].currentClusterIndex] = (Point*)realloc(pointsMat[points[i].currentClusterIndex], clustersSize[points[i].currentClusterIndex] * sizeof(Point));
			pointsMat[points[i].currentClusterIndex][(clustersSize[points[i].currentClusterIndex]) - 1] = points[i];
			
		}
}

double distancePointToPoint(double x1, double y1, double x2, double y2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}

void calClusterCenter(Cluster* cluster, Point* clusterPoints, int clusterPointsSize)//Step 3 in K-Means algorithm
{
	int i;
	double sumX = 0, sumY = 0;

	//Calculate all the cluster points cordinates
	for (int i = 0; i < clusterPointsSize; i++)
	{
		sumX += clusterPoints[i].x;
		sumY += clusterPoints[i].y;
	}

	//Finding the new cluster cordinates(center).
	cluster->x = (sumX / clusterPointsSize);
	cluster->y = (sumY / clusterPointsSize);
}

double calClusterDiameter(Point* clusterPoints, int clusterPointsSize)
{
	int i, j;
	double maxDistance = 0, tempDistance = 0;

	for (i = 0; i < clusterPointsSize; i++)
	{
		for (j = 1; j < clusterPointsSize; j++)
		{
			tempDistance = distancePointToPoint(clusterPoints[i].x, clusterPoints[i].y, clusterPoints[j].x, clusterPoints[j].y);

			if (maxDistance < tempDistance)
			{
				maxDistance = tempDistance;
			}
		}
	}

	return maxDistance;
}

double evaluateQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize)

{
	int i, j;
	double numerator = 0, quality = 0, numOfArguments, currentClustersDistance = 0;

	numOfArguments = K * (K - 1);

#pragma omp parallel for shared(clusters) private(j)
		for (i = 0; i < K; i++)
		{
			//calculate the current cluster's diameter (di) 
			clusters[i].diameter = calClusterDiameter(pointsMat[i], clustersSize[i]);

#pragma omp parallel for
			for (j = 0; j < K; j++)
			{
				if (i != j)
				{
					//calculate the distance between the current cluster and the other clusters (Dij)
					currentClustersDistance = distancePointToPoint(clusters[i].x, clusters[i].y, clusters[j].x, clusters[j].y);

					numerator += clusters[i].diameter / currentClustersDistance;
				}
			}
		}

	// calculate the average of diameters of the cluster divided by distance to other clusters
	quality = numerator / numOfArguments;

	return quality;
}

double kMeansWithIntervals(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	int limit, double QM, double T, double dt, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr,
	int numprocs, double* time, double* currentPointsCordinates, double* initialPointsCordinates, double* pointsVelocityArr)
{
	double n, tempQuality, quality = 0;
	int procsNumber;
	
	for (*time = 0, n = 0; n < T / dt; n++)
	{
		//calculate the current time
		*time = n*dt;
		
#pragma omp parallel for 
		for (procsNumber = 1; procsNumber < numprocs; procsNumber++)
			MPI_Send(time, 1, MPI_DOUBLE, procsNumber, TRANSFER_TAG, MPI_COMM_WORLD);
		
		
		//calculate points cordinates according to current times
		//cuda -> pointsCordinates
		calPointsCordsCuda(*time, initialPointsCordinates, pointsVelocityArr, currentPointsCordinates, numOfPoints*2);
		refreshPointsCordinates(points,numOfPoints,currentPointsCordinates);

		//without
		//calPointsCordinates(points, numOfPoints, *time);
		
		//K-Mean Algorithm
		tempQuality = kMeansAlgorithmMaster(points, clusters, pointsMat, clustersSize, numOfPoints, K,
			limit, numprocs, PointMPIType, ClusterMPIType, sumsArr ,numprocs);
		
		//checks if the quality measure is less than QM
		if (tempQuality < QM)
		{
#pragma omp parallel for 
			for (procsNumber = 1; procsNumber < numprocs; procsNumber++)
				MPI_Send(time, 1, MPI_DOUBLE, procsNumber, FINAL_TERMINATION_TAG, MPI_COMM_WORLD);

			return tempQuality;
		}
			
		//checks if the current given quality measure is better than the best given quality so far.
		if (tempQuality < quality || quality == 0)
			quality = tempQuality;
	}

#pragma omp parallel for 
	for (procsNumber = 1; procsNumber < numprocs; procsNumber++)
		MPI_Send(time, 1, MPI_DOUBLE, procsNumber, FINAL_TERMINATION_TAG, MPI_COMM_WORLD);

	return quality;
}

double kMeansAlgorithmMaster(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	int limit, int numOfProcs, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr, int numprocs)
{
	MPI_Status status;

	int i, j, z, k, flag,tempSize = 0,prevSize, totalFlagCount = 0;
	double* totalSumArr;
	int* totalClustersSize;
	
	totalSumArr = (double*)calloc(K * 2, sizeof(double));
	totalClustersSize = (int*)calloc(K, sizeof(int));

#pragma omp parallel for shared(points)
	for (i = 0; i < numOfPoints; i++)
		points[i].previousClusterIndex = -1;

	for (i = 0; i < limit; i++)
	{
		flag = 0;
		totalFlagCount = 0;

		//Send to the slave the clusters
#pragma omp parallel for 
		for (int procsNumber = 1; procsNumber < numprocs; procsNumber++)
			MPI_Send(clusters, K, ClusterMPIType, procsNumber, TRANSFER_TAG, MPI_COMM_WORLD);
		
		//Step 2 - Group points around the given clusters centers
		groupPointsToClusters(pointsMat, clustersSize, points, numOfPoints, clusters, K);

		//Step 3.1 - computes the total sum of cordiantes x and y in each cluster in the master process
#pragma omp parallel for shared(clusters,pointsMat,clustersSize,sumsArr)
		for (j = 0; j < K; j++)
		{
			calClusterCordSum(clusters + j, pointsMat[j], clustersSize[j], sumsArr + (j * 2), sumsArr + ((j * 2) + 1));
		}

		//Step 3.2 - Obtain the information from the slaves that needed for to computation of the new clusters centers 
		MPI_Reduce(sumsArr, totalSumArr, K * 2, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
		MPI_Reduce(clustersSize, totalClustersSize, K, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

		//Step 3.3 - update the clusters centers
#pragma omp parallel for shared(clusters)
		for (j = 0; j < K; j++)
		{
			clusters[j].x = totalSumArr[(j*2)]/ totalClustersSize[j];
			clusters[j].y = totalSumArr[(j * 2) + 1] / totalClustersSize[j];
		}

		//Step 4 - Checks if some point move to another cluster after the update of clusetrs center cordinates.
		for (j = 0; j < numOfPoints && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
		flag = (j == numOfPoints ? 1 : 0);

		//Send to the master process the answer to the termination condition questsion 
		MPI_Reduce(&flag, &totalFlagCount, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
	
		//Step 5 - Termination condition fulfills
		if (totalFlagCount == numOfProcs)
		{
#pragma omp parallel for 
			for (int procsNumber = 1; procsNumber < numprocs; procsNumber++)
				MPI_Send(clusters, K, ClusterMPIType, procsNumber, MID_TERMINATION_TAG, MPI_COMM_WORLD);

			//Gather all the points from all the process to the Master process
			gatherAllPoints(pointsMat, clustersSize, totalClustersSize, K, numOfProcs, PointMPIType);

			free(totalClustersSize);
			free(totalSumArr);
			return evaluateQuality(pointsMat, clusters, K, clustersSize);
		}

	}
	for (int procsNumber = 1; procsNumber < numprocs; procsNumber++)
		MPI_Send(clusters, K, ClusterMPIType, procsNumber, MID_TERMINATION_TAG, MPI_COMM_WORLD);

	//Gather all the points from all the process to the Master process
	gatherAllPoints(pointsMat, clustersSize, totalClustersSize, K, numOfProcs, PointMPIType);
	
	free(totalClustersSize);
	free(totalSumArr);
	return evaluateQuality(pointsMat, clusters, K, clustersSize);

}

void kMeansAlgorithmSlave(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K , 
	MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr)
{
	MPI_Status status;
	

	status.MPI_TAG = TRANSFER_TAG;
	int j, flag;
	
	while (status.MPI_TAG == TRANSFER_TAG)
	{
		flag = 0;
		
		MPI_Recv(clusters, K, ClusterMPIType, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG == MID_TERMINATION_TAG)
		{
			returnPoints(K, clustersSize, pointsMat, PointMPIType);
		}
		else
		{
			//Step 2 - Group points around the given clusters centers
			groupPointsToClusters(pointsMat, clustersSize, points, numOfPoints, clusters, K);

			//Step 3 -  computes the total sum of cordiantes x and y in each cluster
#pragma omp parallel for shared(clusters,pointsMat,clustersSize,sumsArr)
			for (j = 0; j < K; j++)
			{
				calClusterCordSum(clusters + j, pointsMat[j], clustersSize[j], sumsArr + (j * 2), sumsArr + ((j * 2) + 1));
			}

			MPI_Reduce(sumsArr, sumsArr, K * 2, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
			MPI_Reduce(clustersSize, clustersSize, K, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

			//Step 4 - Checks if some point move to another cluster after the update of clusetrs center cordinates.
			//CUDA
			for (j = 0; j < numOfPoints && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
			flag = (j == numOfPoints ? 1 : 0);

			MPI_Reduce(&flag, &flag, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
		}
		
	}
}

void gatherAllPoints(Point** pointsMat, int* clustersSize, int* totalClustersSize, int K, int numOfProcs, MPI_Datatype PointMPIType)
{
	MPI_Status status;
	int j, z, prevSize, tempSize;

	for (j = 0; j < K; j++)
	{
		pointsMat[j] = (Point*)realloc(pointsMat[j], totalClustersSize[j] * sizeof(Point));

		for (z = 1; z < numOfProcs; z++)
		{
			prevSize = clustersSize[j];
			MPI_Recv(&tempSize, 1, MPI_INT, z, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			clustersSize[j] += tempSize;
			MPI_Recv(pointsMat[j] + prevSize, tempSize, PointMPIType, z, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		}
	}
}

void returnPoints(int K, int* clustersSize, Point** pointsMat, MPI_Datatype PointMPIType)
{
	for (int i = 0; i < K; i++)
	{
		MPI_Send(&(clustersSize[i]), 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(pointsMat[i], clustersSize[i], PointMPIType, MASTER, 0, MPI_COMM_WORLD);
	}
}

void slaveOperation(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int numOfPoints, int K,
	MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType, double* sumsArr, double* currentPointsCordinates, double * initialPointsCordinates, double* pointsVelocityArr)
{
	MPI_Status status;
	status.MPI_TAG = TRANSFER_TAG;
	double time;
	int i;

	while (status.MPI_TAG != FINAL_TERMINATION_TAG)
	{

		MPI_Recv(&time, 1, MPI_DOUBLE, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		if (status.MPI_TAG != FINAL_TERMINATION_TAG)
		{
			//cuda -> pointsCordinates
			calPointsCordsCuda(time, initialPointsCordinates, pointsVelocityArr, currentPointsCordinates, numOfPoints*2);
			refreshPointsCordinates(points,numOfPoints,currentPointsCordinates);

			//without cuda
			//calPointsCordinates(points, numOfPoints, time);

#pragma omp parallel for shared(points)
			for (i = 0; i < numOfPoints; i++)
				points[i].previousClusterIndex = -1;

			kMeansAlgorithmSlave(points, clusters, pointsMat, clustersSize, numOfPoints, K, PointMPIType, ClusterMPIType, sumsArr);
		}
		else
		{
			break;
		}
	}
}

void initPointsInfoArrs(double* currentPointsCordinates, double* initialPointsCordinates, double* pointsVelocityArr, Point* points, int numOfPoints)
{
#pragma omp parallel for shared(currentPointsCordinates,initialPointsCordinates,pointsVelocityArr)
	for (int i = 0; i < numOfPoints; i++)
	{
		currentPointsCordinates[i * 2] = points[i].x;
		currentPointsCordinates[(i * 2) + 1] = points[i].y;
		initialPointsCordinates[i * 2] = points[i].x;
		initialPointsCordinates[(i * 2) + 1] = points[i].y;
		pointsVelocityArr[i * 2] = points[i].vx;
		pointsVelocityArr[(i * 2) + 1] = points[i].vy;
	}
}

void calPointsCordinates(Point* points, int N, double t)
{
	int i;
#pragma omp parallel for shared(points)
	for (i = 0; i < N; i++)
	{
		points[i].x = points[i].x + (t*points[i].vx);
		points[i].y = points[i].y + (t*points[i].vy);
	}
}

void refreshPointsCordinates(Point* points, int numOfPoints, double* currentPointsCordinates)
{
	int i;
#pragma omp parallel for shared(points)
	for (i = 0; i <numOfPoints; i++)
	{
		points[i].x = currentPointsCordinates[(i * 2)];
		points[i].y = currentPointsCordinates[(i * 2) + 1];
	}
}