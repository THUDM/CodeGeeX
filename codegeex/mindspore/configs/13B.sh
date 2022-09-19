#!/bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

CODE_DATA_DIR="" # TODO: set the path to the code data directory

GAS=1

python ${script_dir}/../train.py \
    --distribute true \
    --device_num $RANK_SIZE \
    --sink_size 2 \
    --run_type train \
    --train_and_eval_mode 0 \
    --mode 13B \
    --code_data $CODE_DATA_DIR \
    --param_init_type fp32 \
    --micro_size $GAS \
    --seq_length 2048 \
    --vocab_size 51200 \
    --ckpt_name_prefix code-13B \
    --save_checkpoint=True \
    --save_checkpoint_path /cache/ckpts \
    --save_checkpoint_obs_path <TODO> \ # TODO: set to obs path for saving ckpts
    --save_checkpoint_steps 250 \
    --load_ckpt_path <TODO> \ # TODO: set to obs path for loading ckpt
    --load_ckpt_epoch <TODO> \ # TODO: set to epoch number of loaded ckpt
    --per_batch_size 16 \
    --dropout_rate 0.1 \
    --full_batch 0 \
    --epoch_size 1 \
    --micro_interleaved_size 1 \
    --profiling 0 \
    --tb_dir $LOG_PATH