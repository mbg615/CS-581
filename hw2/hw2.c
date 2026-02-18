/**************************************************************************
*  C program for Homework 2 in CS 581                                     *
*                                                                         *
*  Program simulates Conway's Game of Life to model cellular evolution    *
*  on a dynamic 2D-array using ghost cells for boundary conditions.       *
*  It also measures execution time and checks for stability               *
*  This version of the program uses OpenMP for parallelization.           *
*                                                                         *
*  To Compile: gcc -fopenmp -Wall -O3 -o hw2 hw2.c                        *
*  To run: ./hw2 <size> <iterations> <threads> <output> <optional_seed>   *
*                                                                         *
*  Author: Maddox Guthrie                                                 *
*  Email: mbguthrie1@crimson.ua.edu                                       *
*  Date: February 23, 2026                                                *
***************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

static int T;

/* function to get wall clock time */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

/* function to allocate the 2-D array */
int **allocarray(int N) {
  int *p, **a;

  N += 2;

  p = (int *)malloc(N*N*sizeof(int));
  a = (int **)malloc(N*sizeof(int*));

  if (p == NULL || a == NULL)
    printf("Error allocating memory\n");

  /* for row major sorage */
  for (int i = 0; i < N; i++)
    a[i] = &p[i*N];

  return a;
}

/* function to delete the 2-D array */
void freearray(int **a) {
  free(&a[0][0]);
  free(a);
}

/* function to initialize the zero array */
int **initarray0(int **a, int N) {
  for (int i = 0; i < N + 2; i++)
    for (int j = 0; j < N + 2; j++)
      a[i][j] = 0;

  return a;
}

/* function to initialize the array */
int **initarray(int **a, int N) {
  for (int i = 1; i < N + 1; i++)
    for (int j = 1; j < N + 1; j++)
      a[i][j] = (rand() % 2);

  for(int i = 0; i < N + 2; i++) {
      a[i][0] = 0;
      a[i][N + 1] = 0;
  }

  for(int j = 0; j < N + 2; j++) {
      a[0][j] = 0;
      a[N + 1][j] = 0;
  }

  return a;
}

/* function to print the array */
void printarray(int **a, int N) {
  for (int i = 1; i < N + 1; i++) {
    for (int j = 1; j < N + 1; j++)
      printf("%c ", a[i][j]);
    printf("\n");
  }
}

/* funtion to print the array to a file */
void fprintarray(int **a, int N, FILE *fp) {
  for (int i = 1; i < N + 1; i++) {
    for (int j = 1; j < N + 1; j++)
      fprintf(fp, "%d ", a[i][j]);
    fprintf(fp, "\n");
  }
}

/* function to count the alive cells neighboring a[i][j] */
inline int countAlive(int **a, int i, int j) {
  return a[i - 1][j - 1] + a[i - 1][j] + a[i - 1][j + 1] +
         a[i][j - 1] + a[i][j + 1] + a[i + 1][j - 1] +
         a[i + 1][j] + a[i + 1][j + 1];
}

/* function to compare the last 2 arrays */
inline int compareGenerations(int **a, int **b, int N) {
  N += 2;
  return memcmp(a[0], b[0], N * N * sizeof(int)) == 0;
}

/* function to run the full simulation on the array using 2D parallelism */
int runSimulation2D(int **a, int **b, int N, int M) {
  int final_t = M;
  int stop_early = 0;


  #pragma omp parallel shared(a, b, stop_early, final_t)
  {
    int tid = omp_get_thread_num();
    int P = (int)sqrt(omp_get_num_threads());

    int row_idx = tid / P;
    int col_idx = tid % P;

    int rows_per_thread = N / P;
    int row_rem = N % P;
    int row_start = row_idx * rows_per_thread + (row_idx < row_rem ? row_idx : row_rem) + 1;
    int row_end = row_start + rows_per_thread + (row_idx < row_rem ? 1 : 0);

    int cols_per_thread = N / P;
    int col_rem = N % P;
    int col_start = col_idx * cols_per_thread + (col_idx < col_rem ? col_idx : col_rem) + 1;
    int col_end = col_start + cols_per_thread + (col_idx < col_rem ? 1 : 0);

    for(int t = 0; t < M; t++) {

      for(int i = row_start; i < row_end; i++) {
        for(int j = col_start; j < col_end; j++) {
          int aliveNeighbors = countAlive(a, i, j);
          b[i][j] = (aliveNeighbors == 3) || (aliveNeighbors == 2 && a[i][j]);
        }
      }
      #pragma omp barrier

      #pragma omp single
      {
        if(compareGenerations(a, b, N)) {
          stop_early = 1;
          final_t = t;
        }

        int** tmp = a;
        a = b;
        b = tmp;
      }

      if(stop_early) break;
    }
  }

  return final_t;
}

/* function to run the full simulation on the array using 1D parallelism */
int runSimulation(int **a, int **b, int N, int M) {
  int final_t = M;
  int stop_early = 0;

  #pragma omp parallel shared(a, b, stop_early, final_t)
  {
    for(int t = 0; t < M; t++) {

      #pragma omp for collapse(2)
      for(int i = 1; i < N + 1; i++) {
        for(int j = 1; j < N + 1; j++) {
          int aliveNeighbors = countAlive(a, i, j);
          b[i][j] = (aliveNeighbors == 3) || (aliveNeighbors == 2 && a[i][j]);
        }
      }
      #pragma omp barrier

      #pragma omp single
      {
        if(compareGenerations(a, b, N)) {
          stop_early = 1;
          final_t = t;
        }

        int** tmp = a;
        a = b;
        b = tmp;
      }

      if(stop_early) break;
    }
  }

  return final_t;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    printf("Usage: %s <N> <M> <T> <P>\n", argv[0]);
    exit(-1);
  }

  if(argc == 6) {
    srand(atoi(argv[5]));
  } else srand(time(NULL));

  FILE *fp = fopen(argv[4], "w");
  if(!fp) {
    printf("Could not open file %s", argv[4]);
    exit(-1);
  }

  int N = atoi(argv[1]);
  int M = atoi(argv[2]);
  T = atoi(argv[3]);

  omp_set_num_threads(T);

  int **a = allocarray(N);
  int **b = allocarray(N);

  a = initarray(a, N);
  b = initarray0(b, N);

  double starttime, endtime;

  starttime = gettime();
  int val = runSimulation(a, b, N, M);
  endtime = gettime();

  if(val != M) {
    printf("Simulation finished early after %d iterations.\n", val);
  } else {
    printf("Simulation ran through the maximum of %d iterations.\n", M);
  }

  fprintarray(val % 2 ? a : b, N, fp);

  fclose(fp);
  freearray(a);
  freearray(b);

  printf("Time taken for size %d = %lf seconds\n", N, endtime-starttime);

  return 0;
}
