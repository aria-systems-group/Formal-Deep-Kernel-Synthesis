#!/bin/bash

THREADS=10
EXPNUM=1
export JULIA_NUM_THREADS=$THREADS
#julia test.jl $THREADS
# python3 setup_nn_bounds.py $THREADS $EXPNUM
julia DeepKernelSynthesis.jl $THREADS $EXPNUM 0
