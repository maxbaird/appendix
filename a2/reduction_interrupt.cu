#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#define NUM_ELEMENTS 1<<20
#define BLOCK_SIZE 1024

#define CUDA_ERROR_CHECK(func) { gpuAssert((func), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void reduce(volatile bool *timeout, bool *executedBlocks, int *input, int *output) {

  __shared__ unsigned int block_timeout;

  /*Calculate block ID in grid */
  unsigned long long int bid = blockIdx.x + gridDim.x *
                               (blockIdx.y + gridDim.z * blockIdx.z);

  /* Copy timeout signal from host to local block variable */
  if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    block_timeout = *timeout;
  }

  /* Return if block was previously executed */
  if(executedBlocks[bid]){
    return;
  }

  /* Preventy any warps from proceeding until timeout is copied */
  __syncthreads();

  /* Return if block_timeout is true */
  if(block_timeout){
    return;
  }

  /* Mark block as executed */
  executedBlocks[bid] = true;

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

  volatile bool *timeout = NULL;
  bool complete = false;
  bool *executedBlocks = NULL;

  cudaMallocManaged((void **)&timeout, sizeof(volatile bool), cudaMemAttachGlobal);
  cudaMallocManaged((void **)&executedBlocks, grid_size * sizeof(bool), cudaMemAttachGlobal);

  memset(executedBlocks, 0, grid_size * sizeof(bool));

  *timeout = false;
  size_t interrupt_count = 0;

  CUDA_ERROR_CHECK(cudaMemcpy(deviceInput, hostInput, input_size, cudaMemcpyHostToDevice));

  while(!complete){
    reduce<<<grid_size, BLOCK_SIZE, BLOCK_SIZE*sizeof(int)>>>(timeout, executedBlocks, deviceInput, deviceOutput);
    CUDA_ERROR_CHECK(cudaPeekAtLastError());

    usleep(0.001);
    *timeout = true;
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    /* Check if kernel is complete */
    size_t i = 0;
    for(i = 0; i < grid_size; i++){
      if(executedBlocks[i] == false){
       break;
      } 
    }
    interrupt_count++;

    if(i == grid_size){
      complete = true;
    }else{
      *timeout = false;
    }
  }
  fprintf(stdout, "Interrupt count: %zu\n", interrupt_count);

  CUDA_ERROR_CHECK(cudaMemcpy(hostOutput, deviceOutput, output_size, cudaMemcpyDeviceToHost));

  for(i = 1; i < grid_size; i++){
    hostOutput[0] += hostOutput[i];
  }

  fprintf(stdout, "Result: "); 

  if(hostOutput[0] == NUM_ELEMENTS){
    fprintf(stdout, "PASS\n");
  }else{
    fprintf(stderr, "FAIL\n");
  }

  fprintf(stdout, "Sum = %d\n", hostOutput[0]);

  free(hostInput);
  free(hostOutput);

  CUDA_ERROR_CHECK(cudaFree(deviceInput));
  CUDA_ERROR_CHECK(cudaFree(deviceOutput));
  CUDA_ERROR_CHECK(cudaFree(executedBlocks));
  CUDA_ERROR_CHECK(cudaFree((void *)timeout));

  CUDA_ERROR_CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
