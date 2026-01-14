/**************************************************************************
*  C program for Homework0 in CS 581                                      *
*                                                                         *
*  Program simulates Conway's Game of Life to model cellular evolution    *
*  on a dynamic 2D-array using ghost cells for boundary conditions.       *
*  It also measures execution time and checks for stability               *
*                                                                         *
*  To Compile: gcc -Wall -O -o hw0 hw0.c                                  *
*  To run: ./hw0 <size> <iterations>                                      *
*                                                                         *
*  Author: Maddox Guthrie                                                 *
*  Email: mbguthrie1@crimson.ua.edu                                       *
*  Date: January 21, 2026                                                 *
***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

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

  /* for row major storage */
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
      a[i][j] = rand() % 2;

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
      printf("%i ", a[i][j]);
    printf("\n");
  }
}

/* function to count the alive cells neighboring a[i][j] */
int countAlive(int **a, int i, int j) {
  return a[i - 1][j - 1] + a[i - 1][j] + a[i - 1][j + 1] + a[i][j - 1] + a[i][j + 1] + a[i + 1][j - 1] + a[i + 1][j] + a[i + 1][j + 1];
}

/* function to compare the last 2 arrays */
int compareGenerations(int **a, int **b, int N) {
  N += 2;
  return memcmp(a[0], b[0], N * N * sizeof(int)) == 0;
}

/* function to compute the next generation of the array */
int computeGeneration(int **a, int **b, int N) {
  for(int i = 1; i < N + 1; i++) {
    for(int j = 1; j < N + 1; j++) {
      int aliveNeighbors = countAlive(a, i, j);

      if(a[i][j]) {
        if(aliveNeighbors <= 1) b[i][j] = 0;
        else if(aliveNeighbors >= 4) b[i][j] = 0;
        else b[i][j] = 1;
      } else {
        if(aliveNeighbors == 3) b[i][j] = 1;
        else b[i][j] = 0;
      }
    }
  }

  return compareGenerations(a, b, N);
}

/* function to run the full simulation on the array */
int runSimulation(int **a, int **b, int N, int M) {
  int i;
  for(i = 0; i < M; i++) {
    if(computeGeneration(a, b, N)) break;
    int **tmp = a;
    a = b;
    b = tmp;
  }
  return i;
}

int main(int argc, char **argv) {
  srand(time(NULL));

  double starttime, endtime;
  starttime = gettime();

  int **a=NULL;
  int **b=NULL;

  if (argc != 3) {
    printf("Usage: %s <N> <M>\n", argv[0]);
    exit(-1);
  }

  int N = atoi(argv[1]);
  int M = atoi(argv[2]);

  a = allocarray(N);
  b = allocarray(N);

  a = initarray(a, N);
  b = initarray0(b, N);

  int val = runSimulation(a, b, N, M);
  if(val != M) {
    printf("Simulation finished early after %d iterations.\n", val);
  } else {
    printf("Simulation ran through the maximum of %d iterations.\n", M);
  }

  freearray(a);

  endtime = gettime();
  printf("Time taken for size %d = %lf seconds\n", N, endtime-starttime);

  return 0;
}
