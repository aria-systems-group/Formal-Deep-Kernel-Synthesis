#!/bin/bash

THREADS=10
EXPNUM=6
REFINEMENTS=2
export JULIA_NUM_THREADS=$THREADS
python3 setup_nn_bounds.py $THREADS $EXPNUM
julia DeepKernelSynthesis.jl $THREADS $EXPNUM $REFINEMENTS
