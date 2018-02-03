#!/bin/bash

jsrun -n1 nvprof -s -o single_GPU_naive_UM.%h.%p.nvvp ./run
