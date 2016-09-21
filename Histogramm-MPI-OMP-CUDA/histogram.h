#pragma once

#define ARR_SIZE 512

void GenerateArray(int* arr);
int* OpenMPTask(int* src_arr);
int* CUDATask(int* arr, int size);
void OpenMPFinalMergeTask(int* dest_array, int* src_array);
void PrintArr(int* arr, int size);
void ZeroGivenArray(int* arr, int size);