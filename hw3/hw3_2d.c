/***************************************************************************************
*  C program for Homework 3 in CS 581                                                  *
*                                                                                      *
*  Program simulates Conway's Game of Life to model cellular evolution                 *
*  on a dynamic 2D-array using ghost cells for boundary conditions.                    *
*  It also measures execution time and checks for stability                            *
*  This version of the program uses MPI for parallelization on a 2D grid.              *
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
#include <math.h>
#include <mpi.h>

/* function to allocate the 2-D array */
int **allocarray(int N, int M) {
  int *p  = (int *)malloc(N * M * sizeof(int));
  int **a = (int **)malloc(N * sizeof(int *));

  if(!p || !a) {
    fprintf(stderr, "Error allocating memory\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (int i = 0; i < N; i++) {
    a[i] = &p[i*M];
  }

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

static inline int countAlive(int **a, int i, int j) {
  return a[i - 1][j - 1] + a[i - 1][j] + a[i - 1][j + 1] +
         a[i][j - 1] + a[i][j + 1] + a[i + 1][j - 1] +
         a[i + 1][j] + a[i + 1][j + 1];
}

static inline int compareArrays(int **a, int **b, int rows, int cols) {
    return memcmp(a[0], b[0], rows * cols * sizeof(int)) == 0;
}

static int tileSize(int N, int P, int idx) {
    return N / P + (idx < N % P ? 1 : 0);
}

static int tileOffset(int N, int P, int idx) {
    int base = N / P, rem = N % P;
    return idx * base + (idx < rem ? idx : rem);
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    /* ---- validate that size is a perfect square -------------------- */
    int P = (int)round(sqrt((double)size));
    if (P * P != size) {
        if (rank == 0)
            fprintf(stderr, "Error: number of MPI ranks (%d) must be a perfect square.\n", size);
        MPI_Finalize();
        return -1;
    }

    unsigned int seed;
    if (rank == 0) {
        seed = (argc >= 5) ? (unsigned)atoi(argv[4]) : (unsigned)time(NULL);
    }

    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    srand(seed + (unsigned)rank * 1000003u);

    int dims[2]    = {P, P};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int row_idx = coords[0];
    int col_idx = coords[1];

    int local_rows = tileSize(N, P, row_idx);
    int local_cols = tileSize(N, P, col_idx);

    int row_off = tileOffset(N, P, row_idx);
    int col_off = tileOffset(N, P, col_idx);

    int total_rows = local_rows + 2;
    int total_cols = local_cols + 2;

    int **a = allocarray(total_rows, total_cols);
    int **b = allocarray(total_rows, total_cols);
    zeroarray(a, total_rows, total_cols);
    zeroarray(b, total_rows, total_cols);

    for (int i = 1; i <= local_rows; i++)
        for (int j = 1; j <= local_cols; j++)
            a[i][j] = rand() % 2;

    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);

    int nw_rank = MPI_PROC_NULL, ne_rank = MPI_PROC_NULL;
    int sw_rank = MPI_PROC_NULL, se_rank = MPI_PROC_NULL;

    if (north != MPI_PROC_NULL) {
        int ncoords[2] = {row_idx - 1, col_idx};
        if (col_idx > 0)     { ncoords[1] = col_idx - 1; MPI_Cart_rank(cart_comm, ncoords, &nw_rank); }
        if (col_idx < P - 1) { ncoords[1] = col_idx + 1; MPI_Cart_rank(cart_comm, ncoords, &ne_rank); }
    }
    if (south != MPI_PROC_NULL) {
        int scoords[2] = {row_idx + 1, col_idx};
        if (col_idx > 0)     { scoords[1] = col_idx - 1; MPI_Cart_rank(cart_comm, scoords, &sw_rank); }
        if (col_idx < P - 1) { scoords[1] = col_idx + 1; MPI_Cart_rank(cart_comm, scoords, &se_rank); }
    }

    MPI_Datatype row_type, col_type;
    MPI_Type_contiguous(local_cols, MPI_INT, &row_type);
    MPI_Type_commit(&row_type);

    MPI_Type_vector(local_rows, 1, total_cols, MPI_INT, &col_type);
    MPI_Type_commit(&col_type);

    double starttime = MPI_Wtime();

    int final_t   = S;
    int stop_early = 0;

    for (int t = 0; t < S; t++) {
        MPI_Sendrecv(&a[1][1], 1, row_type, north, 0, &a[local_rows+1][1], 1, row_type, south, 0, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[local_rows][1], 1, row_type, south, 1, &a[0][1], 1, row_type, north, 1, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[1][1], 1, col_type, west, 2, &a[1][local_cols+1], 1, col_type, east, 2, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[1][local_cols], 1, col_type, east, 3, &a[1][0], 1, col_type, west, 3, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[1][1], 1, MPI_INT, nw_rank, 4, &a[local_rows+1][local_cols+1], 1, MPI_INT, se_rank, 4, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[1][local_cols], 1, MPI_INT, ne_rank, 5, &a[local_rows+1][0], 1, MPI_INT, sw_rank, 5, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[local_rows][1], 1, MPI_INT, sw_rank, 6, &a[0][local_cols+1], 1, MPI_INT, ne_rank, 6, cart_comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&a[local_rows][local_cols], 1, MPI_INT, se_rank, 7, &a[0][0], 1, MPI_INT, nw_rank, 7, cart_comm, MPI_STATUS_IGNORE);

        for (int i = 1; i <= local_rows; i++) {
            b[i][0]            = 0;
            b[i][local_cols+1] = 0;
            for (int j = 1; j <= local_cols; j++) {
                int alive = countAlive(a, i, j);
                b[i][j] = (alive == 3) || (alive == 2 && a[i][j]);
            }
        }

        memset(b[0],            0, total_cols * sizeof(int));
        memset(b[local_rows+1], 0, total_cols * sizeof(int));

        int local_same  = compareArrays(a, b, total_rows, total_cols);
        int global_same;
        MPI_Allreduce(&local_same, &global_same, 1, MPI_INT, MPI_LAND, cart_comm);

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

    int tile_size = local_rows * local_cols;
    int *sendbuf  = (int *)malloc(tile_size * sizeof(int));
    for (int i = 0; i < local_rows; i++)
        memcpy(&sendbuf[i * local_cols], &a[i+1][1], local_cols * sizeof(int));

    int *all_tile_sizes = NULL;
    if (rank == 0) all_tile_sizes = (int *)malloc(size * sizeof(int));
    MPI_Gather(&tile_size, 1, MPI_INT, all_tile_sizes, 1, MPI_INT, 0, cart_comm);

    int meta[4] = {row_off, col_off, local_rows, local_cols};
    int *all_meta = NULL;
    if (rank == 0) all_meta = (int *)malloc(4 * size * sizeof(int));
    MPI_Gather(meta, 4, MPI_INT, all_meta, 4, MPI_INT, 0, cart_comm);

    int *displs  = NULL;
    int *recvbuf = NULL;
    if (rank == 0) {
        displs  = (int *)malloc(size * sizeof(int));
        int off = 0;
        for (int r = 0; r < size; r++) { displs[r] = off; off += all_tile_sizes[r]; }
        recvbuf = (int *)malloc(N * N * sizeof(int));
    }

    MPI_Gatherv(sendbuf, tile_size, MPI_INT,
                recvbuf, all_tile_sizes, displs, MPI_INT,
                0, cart_comm);

    if (rank == 0) {
        int *global = (int *)calloc(N * N, sizeof(int));
        int src = 0;
        for (int r = 0; r < size; r++) {
            int rrow_off  = all_meta[r*4 + 0];
            int rcol_off  = all_meta[r*4 + 1];
            int rlrows    = all_meta[r*4 + 2];
            int rlcols    = all_meta[r*4 + 3];
            for (int i = 0; i < rlrows; i++)
                for (int j = 0; j < rlcols; j++)
                    global[(rrow_off + i) * N + (rcol_off + j)] =
                        recvbuf[displs[r] + i * rlcols + j];
            src += all_tile_sizes[r];
        }

        if (stop_early) {
           printf("Simulation finished early after %d iterations.\n", final_t);
        } else {
          printf("Simulation ran through the maximum of %d iterations.\n", S);
        }

        printf("Time taken for size %d = %lf seconds\n", N, endtime - starttime);

        FILE *fp = fopen(outfile, "w");
        if (!fp) { fprintf(stderr, "Could not open file %s\n", outfile); }
        else {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    fprintf(fp, "%d ", global[i * N + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }

        free(global);
        free(all_tile_sizes);
        free(all_meta);
        free(displs);
        free(recvbuf);
    }

    free(sendbuf);
    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
    freearray(a);
    freearray(b);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}
