
#!/usr/bin/env bash

IMAGENET_PATH=$1
nGPUs=$2

## SwiftFormer-XS training
python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --model SwiftFormer_XS --aa="" --mixup 0 --cutmix 0 --data-path "$IMAGENET_PATH" \
--output_dir SwiftFormer_XS_results

## SwiftFormer-S training
python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --model SwiftFormer_S --mixup 0 --cutmix 0 --data-path "$IMAGENET_PATH" \
--output_dir SwiftFormer_S_results

## SwiftFormer-L1 training
python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --model SwiftFormer_L1 --data-path "$IMAGENET_PATH" \
--output_dir SwiftFormer_L1_results

## SwiftFormer-L3 training
python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --model SwiftFormer_L3 --data-path "$IMAGENET_PATH" \
--output_dir SwiftFormer_L3_results
