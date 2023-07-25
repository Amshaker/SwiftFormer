#!/bin/sh
#SBATCH --job-name=swiftformer
#SBATCH --partition=your_partition
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=8000

IMAGENET_PATH=$1
MODEL=$2

srun python main.py --model "$MODEL" \
--data-path "$IMAGENET_PATH" \
--batch-size 128 \
--epochs 300 \


## Note: Disable aa, mixup, and cutmix for SwiftFormer-XS, and disable mixup, and cutmix for SwiftFormer-S.
## By default, this script requests total 16 GPUs on 4 nodes. The batch size per gpu is set to 128,
## tha sums to 128*16=2048 in total.
