# This script is used to convert checkpoint model parallel partitions.

LOAD_CKPT_PATH=$1  # Path to weights in .pt format.
SAVE_CKPT_PATH=$2  # Path to save the output MP checkpoints.
MP_SIZE=$3 # Model parallel size

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"

if [ -z "$MP_SIZE" ]; then
  MP_SIZE=1
fi

# export CUDA settings
export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=0,1


CMD="python $MAIN_DIR/codegeex/megatron/convert_ckpt_parallel.py \
      --load-ckpt-path $LOAD_CKPT_PATH \
      --save-ckpt-path $SAVE_CKPT_PATH \
      --tokenizer-path $TOKENIZER_PATH \
      --target-tensor-model-parallel-size $MP_SIZE \
      --num-layers 39 \
      --hidden-size 5120 \
      --num-attention-heads 40 \
      --max-position-embeddings 2048 \
      --attention-softmax-in-fp32 \
      --fp16 \
      --micro-batch-size 1 \
      --make-vocab-size-divisible-by 52224 \
      --seq-length 2048"

echo "$CMD"
eval "$CMD"