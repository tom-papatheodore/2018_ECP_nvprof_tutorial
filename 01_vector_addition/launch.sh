#!/bin/bash

jsrun -n1 nvprof -s -o test_vecAdd.%h.%p.nvvp ./run
