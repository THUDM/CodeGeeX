# This script is used to translate solutions of HumanEval-X.

LANG_SRC_TYPE=$1  # Source programming language, currently support one of ["python", "java", "cpp", "js", "go"]
LANG_TGT_TYPE=$2    # Target programming language, currently support one of ["python", "java", "cpp", "js", "go"]
OUTPUT_PATH=$3 # Output path of the generated programs.
HOSTLIST=$4    # Provide hostfile if generating distributedly

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"

# export CUDA settings
export CUDA_HOME=/usr/local/cuda-11.1/

# import model configuration
source "$MAIN_DIR/configs/codegeex_13b.sh"

# nccl options
OPTIONS_NCCL="export NCCL_DEBUG=warn; export NCCL_IB_DISABLE=0; export NCCL_IB_GID_INDEX=3"
OPTIONS_PATH="export PATH=$PATH; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
CWD=$(pwd)

# set master ip for zmq server
if [ -z "$HOSTLIST" ]; then
  ZMQ_ADDR=$(hostname -i)
  echo "$ZMQ_ADDR" > "./hostfile"
  HOSTLIST="./hostfile"
else
  ZMQ_ADDR=$(cat $HOSTLIST | head -n 1)
fi
echo "master_ip: $ZMQ_ADDR"

NUM_SAMPLES=1
MICRO_BSZ=1
WORLD_SIZE=1
TEMP=0.8
TOPP=0.95
SEED=42
DATASET=humaneval
TODAY=$(date +%y%m%d)
CHANNEL_PORT=$(expr $RANDOM + 5000)
MASTER_PORT=$(expr $RANDOM + 8000)

# save log file
LOG_DIR=$MAIN_DIR/log
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/$TODAY-translation.log"

if [ -z "$LANG_SRC_TYPE" ]
then
    LANG_SRC_TYPE=python
fi

if [ -z "$LANG_TGT_TYPE" ]
then
    LANG_TGT_TYPE=java
fi

if [ -z "$INPUT_SRC_PATH" ]
then
    INPUT_SRC_PATH=$MAIN_DIR/codegeex/benchmark/humaneval-x/$LANG_SRC_TYPE/data/humaneval_$LANG_SRC_TYPE.jsonl.gz
fi

if [ -z "$INPUT_TGT_PATH" ]
then
    INPUT_TGT_PATH=$MAIN_DIR/codegeex/benchmark/humaneval-x/$LANG_TGT_TYPE/data/humaneval_$LANG_TGT_TYPE.jsonl.gz
fi

if [ -z "$OUTPUT_PATH" ]; then
  OUTPUT_PATH=$MAIN_DIR/codegeex/benchmark/output/humaneval-x/codegeex/
  mkdir -p "$OUTPUT_PATH"
fi

JOB_ID=codegeex-ns$NUM_SAMPLES-t$TEMP-topp$TOPP-seed$SEED-$LANGUAGE

RUN_CMD="python \
  $MAIN_DIR/codegeex/benchmark/humaneval-x/translate_humaneval_x.py \
  --hostfile $HOSTLIST \
  --channel-ip $ZMQ_ADDR \
  --channel-port $CHANNEL_PORT \
  --master-port $MASTER_PORT \
  --tokenizer-path $TOKENIZER_PATH \
  --load-deepspeed \
  --temperature $TEMP \
  --top-p $TOPP \
  --out-seq-length 1024 \
  --micro-batch-size $MICRO_BSZ \
  --samples-per-problem $NUM_SAMPLES \
  --language-src-type $LANG_SRC_TYPE \
  --language-tgt-type $LANG_TGT_TYPE \
  --src-path $INPUT_SRC_PATH \
  --tgt-path $INPUT_TGT_PATH \
  --dataset $DATASET \
  --output-prefix $OUTPUT_PATH/$JOB_ID \
  --gen-node-world-size $WORLD_SIZE \
  --seed $SEED \
  $MODEL_ARGS"

RUN_CMD="$OPTIONS_NCCL; $OPTIONS_PATH; $RUN_CMD"
RUN_CMD="cd $CWD; $RUN_CMD"

if (( WORLD_SIZE != 1 )); then
  RUN_CMD="pdsh -R ssh -w ^$HOSTLIST \"$RUN_CMD\""
fi

echo "$RUN_CMD"
echo "Writing log to $LOG_PATH"
eval "$RUN_CMD" > "$LOG_PATH"
bash $MAIN_DIR/scripts/gather_output.sh $OUTPUT_PATH $JOB_ID 1
