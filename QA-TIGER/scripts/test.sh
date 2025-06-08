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
# bash test.sh /mnt/sda/shenhao/code/QA-TIGER/configs/qa_tiger/vitl14.py 1 /mnt/sda/shenhao/code/QA-TIGER/best/best.pt  /mnt/sda/shenhao/code/QA-TIGER/best.txt
CUDA_VISIBLE_DEVICES=$gpu_idx python /mnt/sda/shenhao/code/QA-TIGER/src/test.py \
    --config $config --mode 'test' \
    --n_experts 7 --topK 7 \
    --weight $weight \
    --output_path $output