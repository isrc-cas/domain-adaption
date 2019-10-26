#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=$((RANDOM + 10000)) --use_env $(dirname "$0")/../train.py \
    --trains "cityscapes_train" \
    --tests "cityscapes_val" "foggy_cityscapes_val" \
    --lr-steps 16 22 \
    --epochs 25 \
    --batch-size 2 \
    --work-dir "works_dir/cityscapes" \
    --lr 1e-5 ${@:2}