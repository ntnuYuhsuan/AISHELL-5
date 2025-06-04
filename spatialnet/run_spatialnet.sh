#!/bin/bash

switch=0

# train from scratch
if switch -eq 0; then
    python ./local/SharedTrainer.py fit \
    --config=configs/SpatialNet.yaml \
    --config=configs/aishell5.yaml \
    --model.channels=[0,1,2,3] \
    --model.arch.dim_input=8 \
    --model.arch.dim_output=8 \
    --model.arch.num_freqs=129 \
    --model.compile=true \
    --data.batch_size=[2,2] \
    --trainer.devices=0,1,2,3,4,5,6,7 \
    --trainer.max_epochs=300 # better performance may be obtained if more epochs are given
    # --trainer.precision=bf16-mixed \
fi

# train from checkpoint
if switch -eq 1; then
    python ./local/SharedTrainer.py fit --config=logs/SpatialNet/version_0/config.yaml \
    --data.batch_size=[2,2] \
    --trainer.devices=4,5,6,7 \
    --ckpt_path=logs/SpatialNet/version_0/checkpoints/epoch135_metric-0.9814.ckpt
fi


python ./local/SharedTrainer.py test --config=logs/SpatialNet/version_1/config.yaml \
 --ckpt_path=logs/SpatialNet/version_1/checkpoints/last.ckpt \
 --trainer.devices=3,4,5,6,7
