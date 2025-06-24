#!/bin/bash
DATA_PATH="~/llama-2-books3"
# DATA_NAME="SaylorTwift/the_pile_books3_minus_gutenberg"

SEQ_LEN=2048
BS=64

GRAD_ACCUM=4 # 256/128

# Experiment details

EXP_DIR=./current_exp
mkdir -p ${EXP_DIR}

#export TTT_IMPLEMENTATION="custom.ttt_layer_nobias_l2reg"

if [ -z "$1" ]; then
        TTT_IMPLEMENTATION="custom.ttt_layer_nobias"
        echo "No TTT implementation specified. Using default: ${TTT_IMPLEMENTATION}"
else
        TTT_IMPLEMENTATION="$1"
        echo "Using TTT implementation: ${TTT_IMPLEMENTATION}"
fi

EXP_NAME="${TTT_IMPLEMENTATION}-linear-125m-books-2k"

# PRETRAINED="/home/zacharie/llm-meta-learning/adaptation/ttt/Test-Time-Training_models"
# cp -r ${PRETRAINED}/${EXP_NAME}  ${EXP_DIR}/${EXP_NAME}
# RESUME_EXP_NAME="ttt-linear-125m-books-2k"

#RESUME_EXP_NAME="${EXP_NAME}"


#debug for previous codebase
#EXP_DIR="/home/zacharie/llm-meta-learning/adaptation/ttt/Test-Time-Training_models"
#EXP_NAME="ttt-linear-125m-books-2k"


LOAD_MODEL_CONFIG='125m-TTT'



function get_update_model_config {
        local use_cache=$1
        echo "dict( use_cache=${use_cache}, mini_batch_size=16, ttt_implementation=\"${TTT_IMPLEMENTATION}\",seq_modeling_block='ttt_linear', ttt_base_lr=0.5)"
        }

# ttt_intermediate_size=768,

# use the function to set the UPDATE_MODEL_CONFIG
UPDATE_MODEL_CONFIG=$(get_update_model_config "False")


export CUDA_VISIBLE_DEVICES=0,1 #2,3,4,5 # 0,1,2,3,
export NCCL_DEBUG=INFO

uv run python3 -m ttt.train  \
        --mesh_dim='!1,-1,1' \
        --dtype='bfloat16' \
        --use_zero_order_training=True \
        --zero_order_frequency=1 \
        --total_steps=8800 \
        --save_checkpoint_freq=1000 \
        --save_milestone_freq=2000 \
        --load_model_config=${LOAD_MODEL_CONFIG} \
        --update_model_config="${UPDATE_MODEL_CONFIG}" \
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


if [ $? -ne 0 ]; then
    echo "Training failed. Exiting script."
    exit 1
fi

echo "Training complete. Now running perplexity evaluation..."

UPDATE_MODEL_CONFIG=$(get_update_model_config "True")


export CUDA_VISIBLE_DEVICES=2

uv run python3 test_perplexity.py  \
        --mesh_dim='!1,1,1' \
        --dtype='bfloat16' \
        --load_model_config=${LOAD_MODEL_CONFIG} \
        --update_model_config="${UPDATE_MODEL_CONFIG}" \
        --exp_dir=${EXP_DIR} \
        --exp_name=${EXP_NAME} \
        --compute_chunk_size=8192

























