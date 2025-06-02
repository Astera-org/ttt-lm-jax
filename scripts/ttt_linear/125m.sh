#!/bin/bash

DATA_PATH="/home/zacharie/llama-2-books3"
# DATA_NAME="SaylorTwift/the_pile_books3_minus_gutenberg"
SEQ_LEN=2048
BS=16

GRAD_ACCUM=16 # 256/16

# Experiment details

EXP_DIR=./current_exp
mkdir -p ${EXP_DIR}


EXP_NAME="ttt-linear-125m-books-2k"

# PRETRAINED="/home/zacharie/llm-meta-learning/adaptation/ttt/Test-Time-Training_models"
# cp -r ${PRETRAINED}/${EXP_NAME}  ${EXP_DIR}/${EXP_NAME}
# RESUME_EXP_NAME="ttt-linear-125m-books-2k"

export CUDA_VISIBLE_DEVICES=0,1,2,3

uv run python3 -m ttt.train  \
        --mesh_dim='!1,-1,1' \
        --dtype='fp32' \
        --total_steps=4800 \
        --save_checkpoint_freq=1000 \
        --save_milestone_freq=2000 \
        --load_model_config='125m-TTT' \
        --update_model_config="dict(seq_modeling_block='ttt_linear', ttt_base_lr=1.0)" \
        --dataset_path=${DATA_PATH} \
        --dataset_name=${DATA_NAME} \
        --seq_length=${SEQ_LEN} \
        --global_batch_size=${BS} \
        --accum_steps=${GRAD_ACCUM} \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME} \
        --resume_exp_name=${RESUME_EXP_NAME} \
        --optimizer.type='adamw' \
        --optimizer.adamw_optimizer.weight_decay=0.1 \
        --optimizer.adamw_optimizer.lr=3e-3 \
        --optimizer.adamw_optimizer.end_lr=1e-5 \
        --optimizer.adamw_optimizer.lr_warmup_steps=480 \
        --optimizer.adamw_optimizer.lr_decay_steps=4800