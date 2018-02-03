#!/bin/bash

jsrun -n1 nvprof -s -o single_GPU_naive.%h.%p.nvvp ./run
