#!/bin/bash
DATA_PATH="~/llama-2-books3"
# DATA_NAME="SaylorTwift/the_pile_books3_minus_gutenberg"

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


EXP_NAME="custom.ttt_layer_nobias_frobenius-linear-125m-books-1k-20250721-031733"

LOAD_MODEL_CONFIG='125m-TTT'

function get_update_model_config {
local use_cache=$1
 echo "dict( use_cache=${use_cache}, mini_batch_size=16, ttt_implementation=\"${TTT_IMPLEMENTATION}\",seq_modeling_block='ttt_linear', ttt_base_lr=0.5)"
 }
# ttt_intermediate_size=768,
# use the function to set the UPDATE_MODEL_CONFIG


export CUDA_VISIBLE_DEVICES=6,7 # 4,5,6,7 #0,1,3,4 #,2,3 #4,5,6,7 #2,3,4,5 # 0,1,2,3,
#export NCCL_DEBUG=INFO



SEQ_LEN=1024


echo "Training complete. Now running perplexity evaluation..."
UPDATE_MODEL_CONFIG=$(get_update_model_config "True")

# Run perplexity evaluation and capture output
PERPLEXITY_OUTPUT_FILE="${EXP_DIR}/${EXP_NAME}/perplexity_results.txt"
uv run python3 test_perplexity.py \
--mesh_dim='!1,1,1' \
--dtype='float32' \
--load_model_config=${LOAD_MODEL_CONFIG} \
--update_model_config="${UPDATE_MODEL_CONFIG}" \
--exp_dir=${EXP_DIR} \
--exp_name=${EXP_NAME} \
--ppl_seq_size=${SEQ_LEN} \
--compute_chunk_size=8192 \
--use_wandb=False \
 2>&1 | tee "${PERPLEXITY_OUTPUT_FILE}"
if [ $? -eq 0 ]; then
 echo "Perplexity evaluation completed successfully."
 echo "Results saved to: ${PERPLEXITY_OUTPUT_FILE}"
else
 echo "Perplexity evaluation failed."
 exit 1
fi