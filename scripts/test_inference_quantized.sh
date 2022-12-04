# This script is used to test the inference of CodeGeeX.

GPU=$1
PROMPT_FILE=$2

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"

# import model configuration
source "$MAIN_DIR/configs/codegeex_13b.sh"

# export CUDA settings
if [ -z "$GPU" ]; then
  GPU=0
fi

export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=$GPU

if [ -z "$PROMPT_FILE" ]; then
  PROMPT_FILE=$MAIN_DIR/tests/test_prompt.txt
fi

# remove --greedy if using sampling
CMD="python $MAIN_DIR/tests/test_inference.py \
        --prompt-file $PROMPT_FILE \
        --tokenizer-path $TOKENIZER_PATH \
        --micro-batch-size 1 \
        --out-seq-length 1024 \
        --temperature 0.2 \
        --top-p 0.95 \
        --top-k 0 \
        --quantize \
        $MODEL_ARGS"

echo "$CMD"
eval "$CMD"
