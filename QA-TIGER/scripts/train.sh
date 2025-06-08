# !/bin/bash

if [[ $# -eq 2 ]] ; then
    config=$1
    gpu_idx=$2
else
    echo 'config=$1 gpu_idx=$2'
    exit 1
fi

CUDA_VISIBLE_DEVICES=$gpu_idx python src/train.py \
    --config $config --mode 'train'
