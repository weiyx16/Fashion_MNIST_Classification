#!/usr/bin/env bash

python launch.py \
    --nproc_per_node "$1" --use_env \
    "$2" $3

