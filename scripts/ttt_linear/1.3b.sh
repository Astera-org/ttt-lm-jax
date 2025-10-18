#!/bin/bash

# Set HuggingFace token (set this in your environment before running)
# export HF_TOKEN="your_token_here"

# Use HuggingFace dataset if no local path provided
DATA_PATH=""
DATA_NAME="SaylorTwift/the_pile_books3_minus_gutenberg"
SEQ_LEN=2048
BS=32

GRAD_ACCUM=8 # 256/256 = 1

# Experiment details

EXP_DIR=./current_exp
mkdir -p ${EXP_DIR}

export TTT_IMPLEMENTATION="custom.ttt_layer_nobias_frobenius"

EXP_NAME="ttt_layer_nobias_frobenius-linear-1.3b-books-2k"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m ttt.train \
        --mesh_dim='!-1,1,1' \
        --dtype='fp32' \
        --total_steps=50000 \
        --save_checkpoint_freq=200 \
        --save_milestone_freq=200 \
        --load_model_config='1b-TTT' \
        --update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0)" \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=1e-3 \
        --optimizer.adamw_optimizer.end_lr=1e-5 \
        --optimizer.adamw_optimizer.lr_warmup_steps=5000 \
        --optimizer.adamw_optimizer.lr_decay_steps=50000 \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME}