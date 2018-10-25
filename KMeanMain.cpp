
#include "KMeansHeader.h"
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
	int totalNumOfPoints, numOfPoints, K, limit;//K - Number of clusters,limit - the maximum number of iterations for K-Mean algorithem.
	double QM, T, dt, quality;
	Point* points;
	Cluster* clusters;
	Point** pointsMat;//Each row I, contains the cluster I points.
	int* clustersSize;//Each array cell I contain the size of the row I in pointsMat.
	double* sumsArr, *currentPointsCordinates, *initialPointsCordinates, *pointsVelocityArr;

	int  namelen, numprocs, myId, i;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;

	//Creation of PointMPIType
	MPI_Datatype PointMPIType;
	creationPointMPIType(&PointMPIType);

	////Creation of ClusterMPIType
	MPI_Datatype ClusterMPIType;
	creationClusterMPIType(&ClusterMPIType);

	if (myId == MASTER)
	{
		//read points from file
		points = readDataFromFile(&totalNumOfPoints, &K, &limit, &QM, &T, &dt);

		//Each process will be charge on numOfPoints points.
		numOfPoints = totalNumOfPoints / numprocs;

		//Choose first K points as the initial clusters centers, Step 1 in K-Means algorithem
		clusters = initClusters(points, K);

		MPI_Bcast(&K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&numOfPoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

		//The master will handle the remainder of points
		numOfPoints += (totalNumOfPoints % numprocs);
	}
	else//Slaves
	{
		MPI_Bcast(&K, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&numOfPoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		//printf("the slave %d got k : %d and numOfPoints : %d\n", myId, K, numOfPoints); fflush(stdout);
	}

	//Create matrix of points, where the number of rows are the number of the clusters.
	pointsMat = (Point**)calloc(K, sizeof(Point*));
	checkAllocation(pointsMat);

	clustersSize = (int*)calloc(K, sizeof(int));
	checkAllocation(clustersSize);

	currentPointsCordinates = (double*)calloc(numOfPoints * 2, sizeof(double));
	checkAllocation(currentPointsCordinates);
	initialPointsCordinates = (double*)calloc(numOfPoints * 2, sizeof(double));
	checkAllocation(initialPointsCordinates);
	pointsVelocityArr = (double*)calloc(numOfPoints * 2, sizeof(double));
	checkAllocation(pointsVelocityArr);

	sumsArr = (double*)calloc(K * 2, sizeof(double));
	checkAllocation(sumsArr);

	if (myId == MASTER)
	{
		double start_time, end_time, time;
		
		for(int procsNumber = 1 ; procsNumber < numprocs ; procsNumber++)
			MPI_Send(points + numOfPoints + ((numOfPoints - (totalNumOfPoints % numprocs)) * (procsNumber - 1)), numOfPoints - (totalNumOfPoints % numprocs), PointMPIType, procsNumber, 0, MPI_COMM_WORLD);

		printf("K-Means Algorithm start now\n"); fflush(stdout);

		start_time = omp_get_wtime();

		initPointsInfoArrs(currentPointsCordinates, initialPointsCordinates, pointsVelocityArr, points, numOfPoints);

		quality = kMeansWithIntervals(points, clusters, pointsMat, clustersSize, numOfPoints, K,
			limit, QM, T, dt,PointMPIType, ClusterMPIType, sumsArr, numprocs,
			&time, currentPointsCordinates, initialPointsCordinates, pointsVelocityArr);

		end_time = omp_get_wtime();

		writeToFile(time, quality, clusters, K);
		printf("The quality is : %lf\n", quality); fflush(stdout);
		printf("Time %g\n", end_time - start_time);
	}
	else//Slave
	{
		points = (Point*)calloc(numOfPoints, sizeof(Point));
		checkAllocation(points);
		clusters = (Cluster*)calloc(K, sizeof(Cluster));
		checkAllocation(clusters);
		

		MPI_Recv(points, numOfPoints, PointMPIType, MASTER, 0, MPI_COMM_WORLD, &status);

		initPointsInfoArrs(currentPointsCordinates, initialPointsCordinates, pointsVelocityArr, points, numOfPoints);

		slaveOperation(points, clusters, pointsMat, clustersSize, numOfPoints, K,
			PointMPIType, ClusterMPIType, sumsArr, currentPointsCordinates , initialPointsCordinates, pointsVelocityArr);
	}

	//Free memory from the heap (dynamic)
	free(clustersSize);
	for (int i = 0; i < K; i++)
	{
		free(pointsMat[i]);
	}
	free(pointsMat);
	free(clusters);
	free(points);
	free(sumsArr);
	free(currentPointsCordinates);
	free(initialPointsCordinates);
	free(pointsVelocityArr);

	//MPI_Barrier(MPI_COMM_WORLD);//All the processes have to wait to the root process and the points initalization.
	printf("bye bye from process %d\n",myId); fflush(stdout);
	MPI_Finalize();
	return 0;
}



