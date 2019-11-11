#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    --master_port=$((RANDOM + 10000)) --use_env $(dirname "$0")/../train.py \
    --trains "voc_2007_trainval" "voc_2012_trainval" \
    --tests "voc_2007_test" "voc_clipart_traintest" \
    --eval-types "voc" \
    --lr-steps 16 22 \
    --epochs 25 \
    --batch-size 1 \
    --num-classes 21 \
    --work-dir "works_dir/voc_2_clipart" \
    --lr 1e-5 ${@:2}