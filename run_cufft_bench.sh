#!/bin/bash

set -e
DEVICE=${1:-6}
HW_SIZE=(32 64 128 224 256 512)
D_SIZE=(10)
BATCH_SIZE=(1 2 4 8 16 32 64 128 256)
N=150

export CUDA_VISIBLE_DEVICE=$DEVICE

for batch_size in ${BATCH_SIZE[*]}
do
  for d_size in ${D_SIZE[*]}
  do
    for hw_size in ${HW_SIZE[*]}
    do
      python fft_bench.py -b $batch_size -d $d_size -hw $hw_size -n $N
    done
  done
done
