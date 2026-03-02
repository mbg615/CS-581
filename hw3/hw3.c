/***************************************************************************************
*  C program for Homework 3 in CS 581                                                  *
*                                                                                      *
*  Program simulates Conway's Game of Life to model cellular evolution                 *
*  on a dynamic 2D-array using ghost cells for boundary conditions.                    *
*  It also measures execution time and checks for stability                            *
*  This version of the program uses MPI for parallelization.                           *
*                                                                                      *
*  To Compile: mpicc -Wall -O3 -o hw3 hw3.c                                            *
*  To run: mpirun -np <processes> ./hw3 <size> <iterations> <output> <optional_seed>   *
*                                                                                      *
*  Author: Maddox Guthrie                                                              *
*  Email: mbguthrie1@crimson.ua.edu                                                    *
*  Date: March 25, 2026                                                                *
****************************************************************************************/

#include <mpi.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

/* function to get wall clock time */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: %s <N> <M> <O>\n", argv[0]);
    exit(-1);
  }

  if(argc == 5) {
    srand(atoi(argv[4]));
  } else srand(time(NULL));

  FILE *fp = fopen(argv[3], "w");
  if(!fp) {
    printf("Could not open file %s", argv[4]);
    exit(-1);
  }

  int N = atoi(argv[1]);
  int M = atoi(argv[2]);

  // ToDo: Rest of main logic

  double starttime, endtime;

  starttime = gettime();
  int val = 0; // ToDo: Run Simulation Function
  endtime = gettime();

  if(val != M) {
    printf("Simulation finished early after %d iterations.\n", val);
  } else {
    printf("Simulation ran through the maximum of %d iterations.\n", M);
  }

  fprintarray(val % 2 ? a : b, N, fp);

  fclose(fp);
  // ToDo: Free any memory

  printf("Time taken for size %d = %lf seconds\n", N, endtime-starttime);

  return 0;
}
