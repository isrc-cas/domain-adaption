#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=2,3

PYTHON=${PYTHON:-"python"}



CUDA_VISIBLE_DEVICES=0,1,2,3 nohup ${PYTHON} -m torch.distributed.launch --nproc_per_node=4 --master_port=$((RANDOM + 10000)) --use_env \
    $(dirname "$0")/../adversarial_train.py \
    --config-file configs/adv_vgg16_cityscapes_2_foggy.yaml \
    --resume work_dir/baseline_cityscapes_2_foggy_align/model_epoch_24.pth