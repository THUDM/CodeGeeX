#!/bin/bash
script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

CODE_DATA_DIR="" # TODO: set the path to the code data directory

GAS=32

python ${script_dir}/../save_1p_ckpt_from_8p_ckpt.py \
    --distribute true \
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
    --save_checkpoint_steps <TODO> \ # TODO: set to epoch number of loaded ckpt
    --load_ckpt_path <TODO> \ # TODO: set to obs path for loading ckpt
    --load_ckpt_epoch <TODO> \ # TODO: set to epoch number of loaded ckpt, same as save_checkpoint_steps
    --strategy_load_ckpt_path "/home/work/user-job-dir/start_1.6/strategy.ckpt" \
    --per_batch_size 16 \
    --full_batch 0 \
    --epoch_size 1 \
    --micro_interleaved_size 1 \
    --profiling 0 \
    --tb_dir $LOG_PATH
