# This script is used to gather the distributed outputs of different ranks.

OUTPUT_DIR=$1
OUTPUT_PREFIX=$2
IF_REMOVE_RANK_FILES=$3

echo "$OUTPUT_DIR"
echo "$OUTPUT_PREFIX"

if [ -z "$IF_REMOVE_RANK_FILES" ]
then
    IF_REMOVE_RANK_FILES=0
fi

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")


CMD="python $MAIN_DIR/codegeex/benchmark/gather_output.py \
        --output_dir $OUTPUT_DIR \
        --output_prefix $OUTPUT_PREFIX \
        --if_remove_rank_files $IF_REMOVE_RANK_FILES"

echo "$CMD"
eval "$CMD"