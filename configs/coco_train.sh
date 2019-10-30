#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=$((RANDOM + 10000)) --use_env $(dirname "$0")/../train.py \
    --num-classes 81 \
    --trains "coco_2017_train" \
    --tests "coco_2017_val" \
    --lr-steps 8 11 \
    --epochs 12 \
    --batch-size 2 \
    --work-dir "works_dir/coco" \
    --lr 1e-5 ${@:2}