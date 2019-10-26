#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1
RESUME=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=$((RANDOM + 10000)) --use_env $(dirname "$0")/../train.py \
    --trains "cityscapes_train" \
    --tests "cityscapes_val" "foggy_cityscapes_val" \
    --eval-types "voc" "coco" \
     --work-dir "works_dir/cityscapes" \
    --test-only --resume=$RESUME ${@:3}