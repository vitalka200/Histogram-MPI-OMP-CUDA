#include <stdlib.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "omp.h"
#include "cuda_histogram.h"
#include "histogram.h"

const int MASTER_RANK = 0;
const int SLAVE_RANK = 1;
const int MAX_WORKERS = 2;

int main(int argc, char* argv[])
{
	int numberOfWorkers, currentId;
	MPI_Status status;
	int* merged_array;
	int* received_array;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &currentId);
	MPI_Comm_size(MPI_COMM_WORLD, &numberOfWorkers);


	if (numberOfWorkers != MAX_WORKERS)
	{
		printf("Program requires %d nodes\n", MAX_WORKERS);
		MPI_Finalize();
		exit(1);
	}

	if (currentId == MASTER_RANK)
	{
		int* initial_array = new int[ARR_SIZE]; ZeroGivenArray(initial_array, ARR_SIZE);
		received_array = new int[ARR_SIZE]; ZeroGivenArray(received_array, ARR_SIZE);
		// Prepare initial matrix
		GenerateArray(initial_array);
		
		printf("Initial Array\n");
		PrintArr(initial_array, ARR_SIZE);

		// send cuda task, perform openmp task, do merge
		MPI_Send(initial_array + ARR_SIZE/2, ARR_SIZE/2, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD);

		merged_array = OpenMPTask(initial_array);

		MPI_Recv(received_array, ARR_SIZE, MPI_INT, SLAVE_RANK, 0, MPI_COMM_WORLD, &status);

		OpenMPFinalMergeTask(merged_array, received_array);
		// Print array after merge
		printf("Histogram Array\n");
		PrintArr(initial_array, ARR_SIZE);
		delete[] initial_array;
	}
	else
	{
		received_array = new int[ARR_SIZE/2];
		// receive cuda task, perform cuda task, send result
		MPI_Recv(received_array, ARR_SIZE/2, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD, &status);
		
		merged_array = CUDATask(received_array, ARR_SIZE/2);

		MPI_Send(merged_array, ARR_SIZE, MPI_INT, MASTER_RANK, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();

	
	delete[] received_array;
	delete[] merged_array;
	return 0;

}


void ZeroGivenArray(int* arr, int size)
{
	for (int i = 0; i < size; i++)
		arr[i] = 0;
}

void GenerateArray(int* arr)
{
	srand(time(NULL));

	for (int i = 0; i < ARR_SIZE; i++)
		arr[i] = rand() % ARR_SIZE;
}

int* OpenMPTask(int* src_arr)
{
	int* dst_arr = new int[ARR_SIZE]; ZeroGivenArray(dst_arr, ARR_SIZE);
	int* tmp_hist;
	#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
		#pragma omp single
		{
			tmp_hist = new int[ARR_SIZE*nthreads];
			ZeroGivenArray(tmp_hist, ARR_SIZE*nthreads);
		}

		#pragma omp for
		for (int i = 0; i < ARR_SIZE/2; i++) // for openmp task we running only on first arr part
			tmp_hist[tid*ARR_SIZE + src_arr[i]]++;

		// merge
		#pragma omp for
		for (int i = 0; i < ARR_SIZE; i++)
			for (int j = 0; j < nthreads; j++)
				dst_arr[i] += tmp_hist[j*ARR_SIZE + i]; // each thread merges specific cell in tmp_arr to dst_arr

	}
	delete[] tmp_hist;
	return dst_arr;
}

int* CUDATask(int* arr, int size)
{
	return calculateHistogramm(arr, size);
}

void OpenMPFinalMergeTask(int* dest_array, int* src_array)
{
	#pragma omp for
	for (int i = 0; i < ARR_SIZE; i++)
		dest_array[i] += src_array[i]; // each thread merges specific cell in tmp_arr to dst_arr
}

void PrintArr(int* arr, int size)
{
	printf("\n=============================\n");
	for (int i = 0; i < size; i++)
	{
		if (i%10 == 0) printf("\n");
		printf("arr[%-5d] = %5d ", i, arr[i]);
	}
	printf("\n=============================\n");
	fflush(stdout);
}