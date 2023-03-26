#!/usr/bin/env bash

IMAGENET_PATH=$1
MODEL=$2
CHECKPOINT=$3
nGPUs=$4

python -m torch.distributed.launch --master_addr="127.0.0.1" --master_port=1234 --nproc_per_node=$nGPUs --use_env main.py --model "$MODEL" \
--resume $CHECKPOINT --eval \
--data-path "$IMAGENET_PATH" \
--output_dir SwiftFormer_test
