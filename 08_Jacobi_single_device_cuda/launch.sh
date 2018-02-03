#!/bin/bash

jsrun -n1 ./set_ulimit.sh nvprof -s -o cuda_singleGPU.%h.%p.nvvp ./run
