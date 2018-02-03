#!/bin/bash

export OMP_NUM_THREADS=4

jsrun -p1 nvprof -s -o omp_multiGPU.%h.%p.nvvp ./run
