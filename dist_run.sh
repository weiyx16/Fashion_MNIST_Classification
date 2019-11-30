#!/usr/bin/env bash

python launch.py \
    --nproc_per_node "$1" --single_gpu_idx "$2" --use_env \
    "$3" $4

