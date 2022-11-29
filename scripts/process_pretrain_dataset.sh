# Process dataset for CodeGeeX pretraining

DATASET_PATH=$1
OUTPUT_PATH=$2
LANGUAGE=$3

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"

if [ -z "$LANGUAGE" ]; then
  LANGUAGE=python
fi

CMD="python $MAIN_DIR/codegeex/data/process_pretrain_dataset.py \
        --dataset_path $DATASET_PATH \
        --tokenizer_path $TOKENIZER_PATH \
        --output_prefix $OUTPUT_PATH \
        --language $LANGUAGE \
        --mode pretrain \
        --seq_len 2048"

echo "$CMD"
eval "$CMD"