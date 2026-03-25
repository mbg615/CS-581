#!/bin/bash

source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.7-gcc12

echo 1D data distribution
echo 5120x5120 @ 5000 w/ 1
mpirun -np 1 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_1_1 12184185
mpirun -np 1 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_1_2 12184185
mpirun -np 1 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_1_3 12184185

echo 5120x5120 @ 5000 w/ 2
mpirun -np 2 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_2_1 12184185
mpirun -np 2 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_2_2 12184185
mpirun -np 2 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_2_3 12184185

echo 5120x5120 @ 5000 w/ 4
mpirun -np 4 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_4_1 12184185
mpirun -np 4 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_4_2 12184185
mpirun -np 4 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_4_3 12184185

echo 5120x5120 @ 5000 w/ 8
mpirun -np 8 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_8_1 12184185
mpirun -np 8 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_8_2 12184185
mpirun -np 8 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_8_3 12184185

echo 5120x5120 @ 5000 w/ 16
mpirun -np 16 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_16_1 12184185
mpirun -np 16 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_16_2 12184185
mpirun -np 16 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_16_3 12184185

echo 5120x5120 @ 5000 w/ 32
mpirun -np 32 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_32_1 12184185
mpirun -np 32 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_32_2 12184185
mpirun -np 32 ./hw3 5120 5000 /scratch/ualclsf0016/o5120_5000_32_3 12184185

echo 2D data distribution
echo 5120x5120 @ 5000 w/ 1x1
mpirun -np 1 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_1x1_1 12184185
mpirun -np 1 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_1x1_2 12184185
mpirun -np 1 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_1x1_3 12184185

echo 5120x5120 @ 5000 w/ 2x2
mpirun -np 4 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_2x2_1 12184185
mpirun -np 4 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_2x2_2 12184185
mpirun -np 4 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_2x2_3 12184185

echo 5120x5120 @ 5000 w/ 3x3
mpirun -np 9 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_3x3_1 12184185
mpirun -np 9 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_3x3_2 12184185
mpirun -np 9 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_3x3_3 12184185

echo 5120x5120 @ 5000 w/ 4x4
mpirun -np 16 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_4x4_1 12184185
mpirun -np 16 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_4x4_2 12184185
mpirun -np 16 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_4x4_3 12184185

echo 5120x5120 @ 5000 w/ 5x5
mpirun -np 25 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_5x5_1 12184185
mpirun -np 25 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_5x5_2 12184185
mpirun -np 25 ./hw3_2D 5120 5000 /scratch/ualclsf0016/o5120_5000_5x5_3 12184185
