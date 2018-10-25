//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//
//typedef struct Cluster
//{
//	double x;
//	double y;
//	double diameter;//The largest distance between 2 points in the current cluster
//};
//
//typedef struct Point
//{
//	double x;
//	double y;
//	double vx;
//	double vy;
//	int currentClusterIndex;
//	int previousClusterIndex;
//};
//
//Point* readDataFromFile(int* totalNumOfPoints, int* K, int* limit, double* QM, double* T, double* dt);
//void checkAllocation(void* pointer);
//Cluster* initClusters(const Point* points, int K);
//int getClosestClusterIndex(double x, double y, Cluster* clusters, int K);
//void groupPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int N, Cluster* clusters, int K);
//double distancePointToPoint(double x1, double y1, double x2, double y2);
//void calClusterCenter(Cluster* cluster, Point* clusterPoints, int clusterPointsSize);
//double evaluateQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize);
//double calClusterDiameter(Point* clusterPoints, int clusterPointsSize);
//void calPointsCordinates(Point* points, int totalNumOfPoints, double t);
//void freeDynamicAllocation(void* pointer);
//double kMeansWithIntervals(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, int limit, double QM, double T, double dt);
//double kMeansAlgorithem(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, int limit);
//
//void main()
//{
//	int i;
//	Point* points;
//	Cluster* clusters;
//	Point** pointsMat;//Each row I, contains the cluster I points.
//	int* clustersSize;//Each array cell I contain the size of the row I in pointsMat.
//	int totalNumOfPoints, K, limit;//K - Number of clusters,limit - the maximum number of iterations for K-Mean algorithem.
//	double QM, T, dt, quality;
//
//
//	read points from file
//	points = readDataFromFile(&totalNumOfPoints, &K, &limit, &QM, &T, &dt);
//	Choose first K points as the initial clusters centers, Step 1 in K-Means algorithem
//
//	clusters = initClusters(points, K);
//
//	Create matrix of points, where the number of rows are the number of the clusters.
//	pointsMat = (Point**)calloc(K, sizeof(Point*));
//	checkAllocation(pointsMat);
//	clustersSize = (int*)calloc(K, sizeof(int));
//	checkAllocation(clustersSize);
//
//	Start of the Algorithem
//	quality = kMeansWithIntervals(points, clusters, pointsMat, clustersSize, totalNumOfPoints, K, limit, QM, T, dt);
//	quality = kMeansAlgorithem(points, clusters, pointsMat, clustersSize, totalNumOfPoints, K, limit);
//
//	Print the result of the K-Means algorithem -> the quality
//	printf("The quality is : %lf\n", quality);
//
//	Free memory from the heap (dynamic)
//	freeDynamicAllocation(clustersSize);
//	for (i = 0; i < K; i++)
//	{
//		freeDynamicAllocation(pointsMat[i]);
//	}
//	free(pointsMat);
//	freeDynamicAllocation(clusters);
//	freeDynamicAllocation(points);
//	
//
//	printf("bye bye\n");
//	system("pause");
//}
//
//void freeDynamicAllocation(void* pointer)
//{
//	free(pointer);
//}
//
//void calPointsCordinates(Point* points, int N, double t)
//{
//	int i;
//	for (i = 0; i < N; i++)
//	{
//		points[i].x = points[i].x + (t*points[i].vx);
//		points[i].y = points[i].y + (t*points[i].vy);
//	}
//}
//
//Point* readDataFromFile(int* totalNumOfPoints, int* K, int* limit, double* QM, double* T, double* dt)
//{
//	int i;
//	const char* POINTS_FILE = "C:\\Users\\mpi\\Documents\\Visual Studio 2015\\Projects\\finalProject\\finalProject\\test.txt";
//	FILE* file = fopen(POINTS_FILE, "r");
//
//	Check if the file exist
//	if (!file)
//	{
//		printf("could not open the file ");
//		system("pause");
//		MPI_Finalize();
//		exit(1);
//	}
//
//	fscanf(file, "%d %d %d %lf %lf %lf\n", totalNumOfPoints, K, limit, QM, T, dt);//Getting the supplied data from input file
//
//	Point* points = (Point*)malloc(*totalNumOfPoints * sizeof(Point));
//	checkAllocation(points);
//
//	Initalize points from file
//	for (i = 0; i < *totalNumOfPoints; i++)
//	{
//		fscanf(file, "%lf %lf %lf %lf\n", &(points[i].x), &(points[i].y), &(points[i].vx), &(points[i].vy));
//		points[i].currentClusterIndex = 0;
//		points[i].previousClusterIndex = -1;
//	}
//
//	fclose(file);
//	return points;
//}
//
//void checkAllocation(void* pointer)
//{
//	if (!pointer)
//	{
//		printf("Dynamic allocation failed\n");
//		exit(1);
//	}
//}
//
//Cluster* initClusters(const Point* points, int K)//Step 1 in K-Means algorithem
//{
//	int i;
//	Cluster* clusters = (Cluster*)malloc(K * sizeof(Cluster));
//	checkAllocation(clusters);
//
//	for (i = 0; i < K; i++)
//	{
//		clusters[i].x = points[i].x;
//		clusters[i].y = points[i].y;
//		clusters[i].diameter = 0;
//	}
//
//	return clusters;
//}
//
//int getClosestClusterIndex(double x, double y, Cluster* clusters, int K)//part of step 2 in K-Means algorithm
//{
//	int i, index = 0;
//	double minDistance, tempDistance;
//
//	minDistance = distancePointToPoint(x, y, clusters[0].x, clusters[0].y);
//
//	for (i = 1; i < K; i++)
//	{
//		tempDistance = distancePointToPoint(x, y, clusters[i].x, clusters[i].y);
//		if (tempDistance < minDistance)
//		{
//			minDistance = tempDistance;
//			index = i;
//		}
//	}
//
//	return index;
//}
//
//void groupPointsToClusters(Point** pointsMat, int* clustersSize, Point* points, int totalNumOfPoints, Cluster* clusters, int K)//Step 2 in K-Means algorithm
//{
//	int i;
//
//	Reset ClustersSize Array Cells
//	for (i = 0; i < K; i++)
//	{
//		clustersSize[i] = 0;
//	}
//
//	finding for each point his closet cluster
//	for (i = 0; i < totalNumOfPoints; i++)
//	{
//		points[i].previousClusterIndex = points[i].currentClusterIndex;
//		points[i].currentClusterIndex = getClosestClusterIndex(points[i].x, points[i].y, clusters, K);
//		clustersSize[points[i].currentClusterIndex]++;
//		pointsMat[points[i].currentClusterIndex] = (Point*)realloc(pointsMat[points[i].currentClusterIndex], clustersSize[points[i].currentClusterIndex] * sizeof(Point));
//		pointsMat[points[i].currentClusterIndex][(clustersSize[points[i].currentClusterIndex]) - 1] = points[i];
//	}
//
//}
//
//double distancePointToPoint(double x1, double y1, double x2, double y2)
//{
//	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
//}
//
//void calClusterCenter(Cluster* cluster, Point* clusterPoints, int clusterPointsSize)//Step 3 in K-Means algorithm
//{
//	int i;
//	double sumX = 0, sumY = 0;
//
//	Calculate all the cluster points cordinates
//	for (int i = 0; i < clusterPointsSize; i++)
//	{
//		sumX += clusterPoints[i].x;
//		sumY += clusterPoints[i].y;
//	}
//
//	Finding the new cluster cordinates(center).
//	cluster->x = (sumX / clusterPointsSize);
//	cluster->y = (sumY / clusterPointsSize);
//}
//
//double calClusterDiameter(Point* clusterPoints, int clusterPointsSize)
//{
//	int i, j;
//	double maxDistance = 0, tempDistance = 0;
//	for (i = 0; i < clusterPointsSize; i++)
//	{
//		for (j = 1; j < clusterPointsSize; j++)
//		{
//			tempDistance = distancePointToPoint(clusterPoints[i].x, clusterPoints[i].y, clusterPoints[j].x, clusterPoints[j].y);
//
//			if (maxDistance < tempDistance)
//			{
//				maxDistance = tempDistance;
//			}
//		}
//	}
//
//	return maxDistance;
//}
//
//double evaluateQuality(Point** pointsMat, Cluster* clusters, int K, int* clustersSize)
//
//{
//	int i, j;
//	double numerator = 0, quality = 0, numOfArguments, currentClustersDistance = 0;
//
//	numOfArguments = K * (K - 1);
//
//	for (i = 0; i < K; i++)
//	{
//		 calculate the current cluster's diameter (di) 
//		clusters[i].diameter = calClusterDiameter(pointsMat[i], clustersSize[i]);
//
//		for (j = 0; j < K; j++)
//		{
//			if (i != j)
//			{
//				calculate the distance between the current cluster and the other clusters (Dij)
//				currentClustersDistance = distancePointToPoint(clusters[i].x, clusters[i].y, clusters[j].x, clusters[j].y);
//
//				numerator += clusters[i].diameter / currentClustersDistance;
//			}
//		}
//	}
//
//	 calculate the average of diameters of the cluster divided by distance to other clusters
//	quality = numerator / numOfArguments;
//
//	return quality;
//}
//
//double kMeansWithIntervals(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, int limit, double QM, double T, double dt)
//{
//	int i;
//	double t, n, tempQuality, quality = 0;
//	match points to clusters
//
//	for (t = 0, n = 0; n < T / dt; n++)
//	{
//		calculate the current time
//		t = n*dt;
//		printf("t = %lf\n", t); fflush(stdout);
//		
//		calculate points cordinates according to current time
//		calPointsCordinates(points, totalNumOfPoints, t);
//
//		K-Mean Algorithm
//		tempQuality = kMeansAlgorithem(points, clusters, pointsMat, clustersSize, totalNumOfPoints, K, limit);
//
//		checks if the quality measure is less than QM
//		if (tempQuality < QM)
//			return tempQuality;
//
//		checks if the current given quality measure is better than the best given quality so far.
//		if (tempQuality < quality || quality == 0)
//			quality = tempQuality;
//	}
//
//	return quality;
//}
//
//double kMeansAlgorithem(Point* points, Cluster* clusters, Point** pointsMat, int* clustersSize, int totalNumOfPoints, int K, int limit)
//{
//	int i, j;
//
//	for (i = 0; i < limit; i++)
//	{
//		Step 2 - Group points around the given clusters centers
//		groupPointsToClusters(pointsMat, clustersSize, points, totalNumOfPoints, clusters, K);
//
//		Step 3 - Recalculate the clusters center
//		for (j = 0; j < K; j++)
//		{
//			calClusterCenter(clusters + j, pointsMat[j], clustersSize[j]);
//		}
//
//		Step 4 - Checks if some point move to another cluster after the update of clusetrs center cordinates.
//		for (j = 0; j < totalNumOfPoints && (points[j].currentClusterIndex == points[j].previousClusterIndex); j++);
//		if (j == totalNumOfPoints)
//			break;
//	}
//	return evaluateQuality(pointsMat, clusters, K, clustersSize);
//}
//
