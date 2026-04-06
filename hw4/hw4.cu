#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>

#define IDX(x,y,N) (((x) * (N)) + (y))

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(-1); \
    } \
}

/* function to allocate the 2-D array */
uint8_t **allocarray(int N, int M) {
    uint8_t *p  = (uint8_t *)malloc(N * M * sizeof(uint8_t));
    uint8_t **a = (uint8_t **)malloc(N * sizeof(uint8_t *));

    if(!p || !a) {
        fprintf(stderr, "Error allocating memory\n");
    }

    for (int i = 0; i < N; i++)
        a[i] = &p[i*M];

    return a;
}

/* function to initialize the array */
uint8_t **initarray(uint8_t **a, int N) {
  for (int i = 1; i < N - 1; i++)
    for (int j = 1; j < N - 1; j++)
      a[i][j] = (rand() % 2);

  for(int i = 0; i < N; i++) {
      a[i][0] = 0;
      a[i][N - 1] = 0;
  }

  for(int j = 0; j < N; j++) {
      a[0][j] = 0;
      a[N - 1][j] = 0;
  }

  return a;
}

/* function to zero initialize the array */
uint8_t **zeroarray(uint8_t **a, int N) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      a[i][j] = 0;

  return a;
}

/* function to delete the 2-D array */
void freearray(uint8_t **a) {
  free(&a[0][0]);
  free(a);
}

/* function to print the array to a file */
void fprintarray(uint8_t **a, int N, FILE *fp) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++)
      fprintf(fp, "%d ", a[i][j]);
    fprintf(fp, "\n");
  }
}

/* device function to count the alive cells neighboring a_d[i][j] */
__device__ int countAlive(const uint8_t *d_a, int i, int j, int N) {
  return d_a[IDX(i-1,j-1,N)] + d_a[IDX(i-1,j,N)] + d_a[IDX(i-1,j+1,N)] +
         d_a[IDX(i,j-1,N)]   +                   d_a[IDX(i,j+1,N)]   + 
         d_a[IDX(i+1,j-1,N)] + d_a[IDX(i+1,j,N)] + d_a[IDX(i+1,j+1,N)];
}

/* kernel to compute the next generation of the array */
__global__ void computeGeneration(uint8_t const *d_a, uint8_t *d_b, int N) {
  int c = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int r = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (r >= N-1 || c >= N-1) return;

  uint8_t n = countAlive(d_a, r, c, N);

  d_b[IDX(r,c,N)] = (n == 3) || (n == 2 && d_a[IDX(r,c,N)]);
}

int main(int argc, char** argv) {
  if(argc < 4) {
    printf("Usage: %s <N> <M> <O>\n", argv[0]);
    exit(-1);
  }

  int N = atoi(argv[1]) + 2; // Boost the size to account for ghost cells.
  int M = atoi(argv[2]);

  FILE *fp = fopen(argv[3], "w");
  if(!fp) {
    printf("Could not open file %s", argv[3]);
    exit(-1);
  }

  if(argc == 5) {
    srand(atoi(argv[4]));
  } else srand(time(NULL));

  uint8_t **a = allocarray(N, N);
  uint8_t **b = allocarray(N, N);

  a = initarray(a, N);
  b = zeroarray(b, N);

  uint8_t *d_a, *d_b;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  CUDA_CHECK(cudaMallocAsync(&d_a, N * N * sizeof(uint8_t), stream1));
  CUDA_CHECK(cudaMallocAsync(&d_b, N * N * sizeof(uint8_t), stream2));

  CUDA_CHECK(cudaMemcpyAsync(d_a, a[0], N*N*sizeof(uint8_t), cudaMemcpyHostToDevice, stream1));
  CUDA_CHECK(cudaMemcpyAsync(d_b, b[0], N*N*sizeof(uint8_t), cudaMemcpyHostToDevice, stream2));
  
  dim3 threads(16, 16);
  dim3 blocks((N-2 + threads.x - 1)/threads.x, (N-2 + threads.y - 1)/threads.y);

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  cudaEventRecord(start);
  for(int i = 0; i < M; i++) {
    computeGeneration<<<blocks, threads>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());

    uint8_t *tmp = d_a;
    d_a = d_b;
    d_b = tmp;
  }
  cudaEventRecord(stop);

  cudaMemcpy(a[0], d_a, N*N*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  fprintarray(a, N, fp);

  cudaFreeAsync(d_a, stream1);
  cudaFreeAsync(d_b, stream2);

  freearray(a);
  freearray(b);

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  cudaEventSynchronize(stop);
  float elapsed = 0;
  cudaEventElapsedTime(&elapsed, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Size: %s x %s | Generations: %s | Time: %.6f seconds\n", argv[1], argv[1], argv[2], elapsed / 1000.0);
  return 0;
}
