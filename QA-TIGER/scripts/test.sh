# !/bin/bash

if [[ $# -eq 4 ]] ; then
    config=$1
    gpu_idx=$2
    weight=$3
    output=$4
else
    echo 'config=$1 gpu_idx=$2 weight=$3 output=$4'
    exit 1
fi

CUDA_VISIBLE_DEVICES=$gpu_idx python src/test.py \
    --config $config --mode 'test' \
    --n_experts 7 --topK 7 \
    --weight $weight \
    --output_path $output