#!/bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

CODE_DATA_DIR="" # TODO: set the path to the code data directory

GAS=32

python ${script_dir}/../generation_humaneval.py \
    --distribute true \
    --device_num $RANK_SIZE \
    --sink_size 2 \
    --run_type predict \
    --train_and_eval_mode 0 \
    --mode 13B \
    --code_data $CODE_DATA_DIR \
    --param_init_type fp32 \
    --micro_size $GAS \
    --seq_length 2048 \
    --vocab_size 51200 \
    --max_generate_length 1024 \
    --ckpt_name_prefix code-13B \
    --save_checkpoint=True \
    --save_checkpoint_path /cache/ckpts \
    --save_checkpoint_obs_path /home \ # TODO: set at will
    --save_checkpoint_steps 99999 \ # TODO: set at will
    --load_ckpt_path <TODO> \ # TODO: set to obs path for loading ckpt
    --load_ckpt_epoch <TODO> \ # TODO: set to epoch number of loaded ckpt, same as save_checkpoint_steps
    --per_batch_size 6 \
    --full_batch 1 \
    --epoch_size 1 \
    --micro_interleaved_size 1 \
    --profiling 0 \
    --use_past "true" \
    --top_p 0.95 \
    --top_k_num 100 \
    --temperature 0.8 \
    --frequency_penalty 0.0 \
    --presence_penalty 0.0 \
    --strategy_load_ckpt_path "/home/work/user-job-dir/start_1.6/strategy.ckpt" \
    --tb_dir $LOG_PATH \
    --part $PART
    
