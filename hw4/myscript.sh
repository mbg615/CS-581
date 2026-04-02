#! /bin/bash

source /apps/profiles/modules_asax.sh.dyn
module load cuda

./hw4 5120 5000 /scratch/ualclsf0016/o5120_5000_1 12184185
./hw4 5120 5000 /scratch/ualclsf0016/o5120_5000_2 12184185
./hw4 5120 5000 /scratch/ualclsf0016/o5120_5000_3 12184185
