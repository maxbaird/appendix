#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define MASTER 0

#define CUDA_ERROR_CHECK(fun)                                         \
do{                                                                   \
    cudaError_t err = fun;                                            \
    if(err != cudaSuccess){                                           \
      fprintf(stderr, "Cuda error :: %s\n", cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
}while(0);

#define KERNEL_ERROR_CHECK()                                          \
do{                                                                   \
  cudaError_t err = cudaGetLastError();                               \
  if (err != cudaSuccess){                                            \
    fprintf(stderr, "Kernel error :: %s\n", cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                                               \
  }                                                                   \
}while(0);

#define ITERATIONS 1e5

__global__ void 
vector_add(const unsigned short int *a, 
           const unsigned short int *b, 
           unsigned short int *c, 
           unsigned long long n){

  /* Get our global thread ID */
  unsigned long long id = blockIdx.x*blockDim.x+threadIdx.x;

  if(id > n){
    return;
  }

  for(size_t i = 0; i < ITERATIONS; i++){ 
    c[id] = a[id] + b[id];
  }
}

int main(int argc, char *argv[]){
  if(argc != 2){
    fprintf(stderr, "Usage: %s <vector-size\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  
  int rank = 0;
  int processes = 0;
  double start = 0.0;
  double end = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  fprintf(stdout, "Number of processes: %d\n", processes);
  fflush(stdout);

  unsigned long long vector_size = strtoull(argv[1], NULL, 10);
  size_t size = vector_size * sizeof(unsigned short int);

  unsigned short int *h_a = (unsigned short int *)malloc(size);
  unsigned short int *h_b = (unsigned short int *)malloc(size);
  unsigned short int *h_c = (unsigned short int *)malloc(size);

  if(h_a == NULL || h_b == NULL || h_c == NULL){
    fprintf(stderr, "Failed to allocate %zu bytes!\n", size);
    exit(EXIT_FAILURE);
  }

  unsigned short int *d_a = NULL;
  unsigned short int *d_b = NULL;
  unsigned short int *d_c = NULL;

  unsigned long long i = 0;
  unsigned long long j = 0;
  unsigned long long local_sum = 0;

  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_a, size));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_b, size));
  CUDA_ERROR_CHECK(cudaMalloc((void **)&d_c, size));

  /* Initialize vectors */
  for(i = 0; i < vector_size; i++){
    h_a[i] = 1;
    h_b[i] = 1; 
    h_c[i] = 0;
  } 

  CUDA_ERROR_CHECK(cudaMemcpy((void *)d_a, (const void*)h_a, size, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy((void *)d_b, (const void*)h_b, size, cudaMemcpyHostToDevice));
 
  unsigned long long block_size = 1024; 
  unsigned long long grid_size = (unsigned long long)(ceill((long double)vector_size/(long double)block_size));

  cudaEvent_t knl_start, knl_stop;
  cudaEventCreate(&knl_start);
  cudaEventCreate(&knl_stop);

  cudaEventRecord(knl_start);
  vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, vector_size);
  KERNEL_ERROR_CHECK();
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  CUDA_ERROR_CHECK(cudaMemcpy((void *)h_c, (const void *)d_c, size, cudaMemcpyDeviceToHost));
  cudaEventRecord(knl_stop);

  cudaEventSynchronize(knl_stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, knl_start, knl_stop);

  fprintf(stdout, "%d kernel = %fms\n", processes, milliseconds);

  for(j = 0; j < vector_size; j++){
    local_sum = local_sum + h_c[j];
  }

  if(rank == MASTER){
    unsigned long long global_sum = local_sum;
    int i = 0;
    for(i = 1; i < processes; i++){
      MPI_Recv(&local_sum, 1, MPI_UNSIGNED_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      global_sum = global_sum + local_sum;
    }

    if((2 * vector_size * processes) == global_sum){
      fprintf(stdout, "Result: Pass\n");
    }
    else{
      fprintf(stderr, "Result: Failed\n");
    }

    fprintf(stdout, "Sum: %llu\n", global_sum);
  }
  else{
    MPI_Send(&local_sum, 1, MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_COMM_WORLD);
  }

  /* Housekeeping... */
  free(h_a);
  free(h_b);
  free(h_c);
  
  CUDA_ERROR_CHECK(cudaFree((void *)d_a));
  CUDA_ERROR_CHECK(cudaFree((void *)d_b));
  CUDA_ERROR_CHECK(cudaFree((void *)d_c));

  CUDA_ERROR_CHECK(cudaDeviceReset());

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  MPI_Finalize();

  if(rank == MASTER){
    fprintf(stdout, "Runtime = %f secs\n", end - start);
  }

  return EXIT_SUCCESS;
}
