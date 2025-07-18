#!/bin/bash
DATA_PATH="~/llama-2-books3"
# DATA_NAME="SaylorTwift/the_pile_books3_minus_gutenberg"
SEQ_LEN=2048
BS=128
GRAD_ACCUM=2 # 256/128
# Experiment details
EXP_DIR=./current_exp
mkdir -p ${EXP_DIR}
#export TTT_IMPLEMENTATION="custom.ttt_layer_nobias_l2reg"
if [ -z "$1" ]; then
 TTT_IMPLEMENTATION="custom.ttt_layer_nobias_frobenius"
 echo "No TTT implementation specified. Using default: ${TTT_IMPLEMENTATION}"
else
 TTT_IMPLEMENTATION="$1"
 echo "Using TTT implementation: ${TTT_IMPLEMENTATION}"
fi
EXP_NAME="${TTT_IMPLEMENTATION}-linear-125m-books-1k"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
EXP_NAME="${EXP_NAME}-${TIMESTAMP}"
LOAD_MODEL_CONFIG='125m-TTT'

#RESUME_EXP_NAME="custom.ttt_layer_nobias_frobenius-linear-125m-books-2k-20250715-054449"

function get_update_model_config {
local use_cache=$1
 echo "dict( use_cache=${use_cache}, mini_batch_size=16, ttt_implementation=\"${TTT_IMPLEMENTATION}\",seq_modeling_block='ttt_linear', ttt_base_lr=0.5)"
 }
# ttt_intermediate_size=768,
# use the function to set the UPDATE_MODEL_CONFIG


UPDATE_MODEL_CONFIG=$(get_update_model_config "False")
export CUDA_VISIBLE_DEVICES=3,4 #2,5,6,7 #0,1,2,3 #4,5,6,7 #2,3,4,5 # 0,1,2,3,
export NCCL_DEBUG=INFO



uv run python3 -m ttt.train \
--mesh_dim='!1,-1,1' \
--dtype='bfloat16' \
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
--optimizer.adamw_optimizer.lr_decay_steps=8800 \
--zero_order_perturbation_scale=1e-3 \
--use_zero_order_training=False \
--zero_order_num_perturbations=64 \
--zero_order_start_step=4800 \
--zero_order_frequency=1 \
--zero_order_debug_cosine=False



if [ $? -ne 0 ]; then
 echo "Training failed. Exiting script."
 exit 1
fi

SEQ_LEN=1024


echo "Training complete. Now running perplexity evaluation..."
UPDATE_MODEL_CONFIG=$(get_update_model_config "True")
export CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
# Run perplexity evaluation and capture output
PERPLEXITY_OUTPUT_FILE="${EXP_DIR}/${EXP_NAME}/perplexity_results.txt"
uv run python3 test_perplexity.py \
--mesh_dim='!1,1,1' \
--dtype='bfloat16' \
--load_model_config=${LOAD_MODEL_CONFIG} \
--update_model_config="${UPDATE_MODEL_CONFIG}" \
--exp_dir=${EXP_DIR} \
--exp_name=${EXP_NAME} \
--ppl_seq_size=${SEQ_LEN} \
--compute_chunk_size=8192 \
 2>&1 | tee "${PERPLEXITY_OUTPUT_FILE}"
if [ $? -eq 0 ]; then
 echo "Perplexity evaluation completed successfully."
 echo "Results saved to: ${PERPLEXITY_OUTPUT_FILE}"
else
 echo "Perplexity evaluation failed."
 exit 1
fi