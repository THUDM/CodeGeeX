# This script is used to run gradio server for CodeGeeX.

GPU=$1

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"
EXAMPLE_PATH="$MAIN_DIR/deployment/example_inputs.jsonl"

# import model configuration
source "$MAIN_DIR/configs/codegeex_13b.sh"

# export CUDA settings
if [ -z "$GPU" ]; then
  GPU=0
fi

export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=$GPU

# remove --greedy if using sampling
CMD="python $MAIN_DIR/deployment/server_gradio.py \
        --tokenizer-path $TOKENIZER_PATH \
        --example-path $EXAMPLE_PATH \
        $MODEL_ARGS"

echo "$CMD"
eval "$CMD"