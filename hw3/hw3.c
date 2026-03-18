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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

/* function to allocate the 2-D array */
int **allocarray(int N, int M) {
    int *p  = (int *)malloc(N * M * sizeof(int));
    int **a = (int **)malloc(N * sizeof(int *));

    if(!p || !a) {
        fprintf(stderr, "Error allocating memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < N; i++)
        a[i] = &p[i*M];

    return a;
}

/* function to delete the 2-D array */
void freearray(int **a) {
    free(a[0]);
    free(a);
}

/* function to zero the 2-D array */
void zeroarray(int **a, int rows, int cols) {
    memset(a[0], 0, rows * cols * sizeof(int));
}

/* function to count the alive cells neighboring a[i][j] */
static inline int countAlive(int **a, int i, int j) {
    return a[i - 1][j - 1] + a[i - 1][j] + a[i - 1][j + 1] +
           a[i][j - 1] + a[i][j + 1] + a[i + 1][j - 1] +
           a[i + 1][j] + a[i + 1][j + 1];
}

/* function to compare arrays */
static inline int compareArrays(int **a, int **b, int rows, int cols) {
    return memcmp(a[0], b[0], rows * cols * sizeof(int)) == 0;
}

/* main function */
int main(int argc, char **argv) {
    int rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <N> <M> <output> [seed]\n", argv[0]);
        }

        MPI_Finalize();
        return -1;
    }

    int N = atoi(argv[1]);
    int S = atoi(argv[2]);
    const char *outfile = argv[3];

    unsigned int seed;
    if (rank == 0) {
        seed = (argc >= 5) ? (unsigned)atoi(argv[4]) : (unsigned)time(NULL);
    }

    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    srand(seed + (unsigned)rank * 1000003u);

    int base = N / comm_sz;
    int rem  = N % comm_sz;

    int local_rows = base + (rank < rem ? 1 : 0);
    int row_offset = rank * base + (rank < rem ? rank : rem);
    int cols = N + 2;
    int total_rows = local_rows + 2;

    int **a = allocarray(total_rows, cols);
    int **b = allocarray(total_rows, cols);
    zeroarray(a, total_rows, cols);
    zeroarray(b, total_rows, cols);

    for (int i = 1; i <= local_rows; i++)
        for (int j = 1; j <= N; j++)
            a[i][j] = rand() % 2;

    int up_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down_rank = (rank == comm_sz - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Datatype row_type;
    MPI_Type_contiguous(N, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    double starttime = MPI_Wtime();

    int final_t  = S;
    int stop_early = 0;

    for (int t = 0; t < S; t++) {
        MPI_Sendrecv(&a[1][1], 1, row_type, up_rank, 0, &a[local_rows+1][1],  1, row_type, down_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[local_rows][1], 1, row_type, down_rank, 1, &a[0][1], 1, row_type, up_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= N; j++) {
                int alive = countAlive(a, i, j);
                b[i][j] = (alive == 3) || (alive == 2 && a[i][j]);
            }

            b[i][0]   = 0;
            b[i][N+1] = 0;
        }

        memset(b[0],             0, cols * sizeof(int));
        memset(b[local_rows+1],  0, cols * sizeof(int));

        int local_same = compareArrays(a, b, total_rows, cols);
        int global_same;
        MPI_Allreduce(&local_same, &global_same, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        int **tmp = a;
        a = b;
        b = tmp;

        if (global_same) {
            stop_early = 1;
            final_t = t;
            break;
        }
    }

    double endtime = MPI_Wtime();

    int *sendbuf = (int *)malloc(local_rows * N * sizeof(int));
    for (int i = 0; i < local_rows; i++)
        memcpy(&sendbuf[i * N], &a[i+1][1], N * sizeof(int));

    int *recvcounts = NULL;
    int *displs = NULL;
    int *recvbuf = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc(comm_sz * sizeof(int));
        displs = (int *)malloc(comm_sz * sizeof(int));
        int offset = 0;
        for (int r = 0; r < comm_sz; r++) {
            int rrows = base + (r < rem ? 1 : 0);
            recvcounts[r] = rrows * N;
            displs[r] = offset;
            offset += rrows * N;
        }
        recvbuf = (int *)malloc(N * N * sizeof(int));
    }

    MPI_Gatherv(sendbuf, local_rows * N, MPI_INT, recvbuf, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (stop_early) {
          printf("Simulation finished early after %d iterations.\n", final_t);
        } else {
          printf("Simulation ran through the maximum of %d iterations.\n", S);
        }

        printf("Time taken for size %d = %lf seconds\n", N, endtime - starttime);

        FILE *fp = fopen(outfile, "w");
        if (!fp) {
          fprintf(stderr, "Could not open file %s\n", outfile);
        } else {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++)
                    fprintf(fp, "%d ", recvbuf[i * N + j]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
        free(recvcounts);
        free(displs);
        free(recvbuf);
    }

    free(sendbuf);
    MPI_Type_free(&row_type);
    freearray(a);
    freearray(b);

    MPI_Finalize();
    return 0;
}
