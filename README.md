Sequential:
gcc sequential.c helper.c -o seq -lm

OpenMP:
gcc reg_omp.c helper.c -o omp -lm -fopenmp
