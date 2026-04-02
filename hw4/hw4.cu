#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <cuda_device_runtime_api.h>

#define IDX(x,y,N) (((x) * (N)) + (y))

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(-1); \
    } \
}

/* function to get wall clock time */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

/* function to allocate the 2-D array */
int **allocarray(int N, int M) {
    int *p  = (int *)malloc(N * M * sizeof(int));
    int **a = (int **)malloc(N * sizeof(int *));

    if(!p || !a) {
        fprintf(stderr, "Error allocating memory\n");
    }

    for (int i = 0; i < N; i++)
        a[i] = &p[i*M];

    return a;
}

/* function to initialize the array */
int **initarray(int **a, int N) {
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
int **zeroarray(int **a, int N) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      a[i][j] = 0;

  return a;
}

/* function to delete the 2-D array */
void freearray(int **a) {
  free(&a[0][0]);
  free(a);
}

/* function to print the array to a file */
void fprintarray(int **a, int N, FILE *fp) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++)
      fprintf(fp, "%d ", a[i][j]);
    fprintf(fp, "\n");
  }
}

/* device function to count the alive cells neighboring a_d[i][j] */
__device__ int countAlive(int *d_a, int i, int j, int N) {
  return d_a[IDX(i-1,j-1,N)] + d_a[IDX(i-1,j,N)] + d_a[IDX(i-1,j+1,N)] +
         d_a[IDX(i,j-1,N)]   +                   d_a[IDX(i,j+1,N)]   + 
         d_a[IDX(i+1,j-1,N)] + d_a[IDX(i+1,j,N)] + d_a[IDX(i+1,j+1,N)];
}

/* kernel to compute the next generation of the array */
__global__ void computeGeneration(int *d_a, int *d_b, int N) {
  int c = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int r = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (r >= N-1 || c >= N-1) return;

  int n = countAlive(d_a, r, c, N);

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

  int **a = allocarray(N, N);
  int **b = allocarray(N, N);

  a = initarray(a, N);
  b = zeroarray(b, N);

  int *d_a, *d_b;

  double starttime, endtime;
  starttime = gettime();

  CUDA_CHECK(cudaMalloc(&d_a, N * N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_b, N * N * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_a, a[0], N*N*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b[0], N*N*sizeof(int), cudaMemcpyHostToDevice));
  
  dim3 threads(16, 16);
  dim3 blocks((N-2 + threads.x - 1)/threads.x, (N-2 + threads.y - 1)/threads.y);

  for(int i = 0; i < M; i++) {
    computeGeneration<<<blocks, threads>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());

    int *tmp = d_a;
    d_a = d_b;
    d_b = tmp;
  }

  cudaDeviceSynchronize();
  endtime = gettime();

  cudaMemcpy(a[0], d_a, N*N*sizeof(int), cudaMemcpyDeviceToHost);
  fprintarray(a, N, fp);

  freearray(a);
  freearray(b);

  cudaFree(d_a);
  cudaFree(d_b);

  float elapsed = endtime - starttime;
  printf("Size: %s x %s | Generations: %s | Time: %.6f seconds\n", argv[1], argv[1], argv[2], elapsed);
  
  return 0;
}

