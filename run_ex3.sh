#!/bin/bash

THREADS=10
EXPNUM=3
REFINEMENTS=3
export JULIA_NUM_THREADS=$THREADS
#python3 setup_nn_bounds.py $THREADS $EXPNUM
julia DeepKernelSynthesis.jl $THREADS $EXPNUM $REFINEMENTS
