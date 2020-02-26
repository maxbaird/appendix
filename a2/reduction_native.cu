#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_ELEMENTS 1<<30
#define BLOCK_SIZE 1024

#define CUDA_ERROR_CHECK(fun)                                        \
do{                                                                  \
    cudaError_t err = fun;                                           \
    if(err != cudaSuccess){                                          \
      fprintf(stderr, "Cuda error:: %s\n", cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
}while(0);

__global__ void reduce(int *input, int *output) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = input[i];

  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }

    __syncthreads();
  }

  if (tid == 0){
    output[blockIdx.x] = sdata[0];
  }
}

int main(){
  size_t elems = NUM_ELEMENTS;
  size_t grid_size = (size_t)(ceill((long double)elems/(long double)BLOCK_SIZE)); 

  size_t input_size = elems * sizeof(int);
  size_t output_size = grid_size * sizeof(int);

  int *deviceInput = NULL;
  int *deviceOutput = NULL;
  int *hostInput = NULL; 
  int *hostOutput = NULL; 

  hostInput = (int *)malloc(input_size);
  hostOutput = (int *)malloc(output_size);

  if(hostInput == NULL){
    fprintf(stderr, "Failed to allocate %zu bytes for input!\n", input_size);
    exit(EXIT_FAILURE);
  }

  if(hostOutput == NULL){
    fprintf(stderr, "Failed to allocate %zu bytes for output!\n", output_size);
    exit(EXIT_FAILURE);
  }

  CUDA_ERROR_CHECK(cudaMalloc((void **)&deviceInput, input_size));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&deviceOutput, output_size));

  size_t i = 0;

  for(i = 0; i < elems; i++){
      hostInput[i] = 1;
  }

  CUDA_ERROR_CHECK(cudaMemcpy(deviceInput, hostInput, input_size, cudaMemcpyHostToDevice));

  reduce<<<grid_size, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(deviceInput, deviceOutput);

  CUDA_ERROR_CHECK(cudaMemcpy(hostOutput, deviceOutput, output_size, cudaMemcpyDeviceToHost));

  for(i = 1; i < grid_size; i++){
    hostOutput[0] += hostOutput[i];
  }

  fprintf(stdout, "Sum = %d\n", hostOutput[0]);

  free(hostInput);
  free(hostOutput);

  CUDA_ERROR_CHECK(cudaFree(deviceInput));
  CUDA_ERROR_CHECK(cudaFree(deviceOutput));
  CUDA_ERROR_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}
