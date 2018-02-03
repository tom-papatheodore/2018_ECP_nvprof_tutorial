#!/bin/bash

jsrun -n1 nvprof -s -o single_GPU.%h.%p.nvvp ./run
