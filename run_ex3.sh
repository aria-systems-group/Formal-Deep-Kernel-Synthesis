#!/bin/bash

#python3 setup_nn_bounds.py 10 999
START=1
END=3
for (( i=$START; i<=$END; i++ ))
do
  julia GP_bounding.jl 10 999 "$i"
  if [ $i -lt $END ]
  then
    python3 generate_imdp.py 10 999 "$i" 1
  else
    python3 generate_imdp.py 10 999 "$i" 0
  fi
done