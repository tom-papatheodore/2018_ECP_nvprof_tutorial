#!/bin/bash

jsrun -n1 -a4 -c20 ./set_ulimit.sh nvprof -s -o mpi_multiGPU.%h.%{PMIX_RANK}.nvvp ./run
