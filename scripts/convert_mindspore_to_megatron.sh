# This script is used to convert mindspore checkpoint to the megatron format.

NPY_CKPT_PATH=$1  # Path to Mindspore exported weights in .npy format.
SAVE_CKPT_PATH=$2  # Path to save the output .pt checkpoint.
GPU=$3

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"

# export CUDA settings
if [ -z "$GPU" ]; then
  GPU=0
fi

export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=$GPU


CMD="python $MAIN_DIR/codegeex/megatron/mindspore_to_megatron.py \
      --npy-ckpt-path $NPY_CKPT_PATH \
      --save-ckpt-path $SAVE_CKPT_PATH \
      --tokenizer-path $TOKENIZER_PATH \
      $MODEL_ARGS"

echo "$CMD"
eval "$CMD"