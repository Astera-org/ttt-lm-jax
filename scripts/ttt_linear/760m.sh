#!/bin/bash

# Set HuggingFace token (set this in your environment before running)
# export HF_TOKEN="your_token_here"

# Use HuggingFace dataset if no local path provided
DATA_PATH=""
DATA_NAME="SaylorTwift/the_pile_books3_minus_gutenberg"
SEQ_LEN=2048
BS=64

GRAD_ACCUM=4 # 256/256 = 1


EXP_DIR=./current_exp
mkdir -p ${EXP_DIR}

export TTT_IMPLEMENTATION="custom.ttt_layer_nobias_frobenius"

EXP_NAME="ttt_layer_nobias_frobenius-linear-760m-books-2k"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python3 -m ttt.train \
        --mesh_dim='!-1,1,1' \
        --dtype='fp32' \
        --total_steps=29000 \
        --save_checkpoint_freq=200 \
        --save_milestone_freq=200 \
        --load_model_config='760m-TTT' \
        --update_model_config="dict(ttt_implementation=\"${TTT_IMPLEMENTATION}\",seq_modeling_block='ttt_linear', ttt_base_lr=1.0)" \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=1.25e-3 \
        --optimizer.adamw_optimizer.end_lr=1e-5 \
        --optimizer.adamw_optimizer.lr_warmup_steps=2900 \
        --optimizer.adamw_optimizer.lr_decay_steps=29000 \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME}