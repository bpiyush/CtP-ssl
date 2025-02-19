#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_net.py --cfg $CONFIG --launcher pytorch ${@:3} \
    --data_dir /ssd/pbagad/datasets/ \
    --freeze_backbone \
    --ckpt /home/pbagad/projects/ssl_benchmark/checkpoints/CTP/Kinetics/pretext_checkpoint/r2p1d18_ctp_k400_epoch_90.pth