import os
import torch
import logging

logging.getLogger("torch").setLevel(logging.WARNING)

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from functools import partial

from codegeex.megatron import get_args, print_rank_0, get_timers,get_tokenizer, mpu
from codegeex.megatron.data.prompt_dataset import build_train_valid_test_datasets
from codegeex.megatron.model import CodeGeeXModel
from codegeex.megatron.training import pretrain
from codegeex.megatron.utils import get_ltor_masks_and_position_ids
from codegeex.megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0("building GPT model ...")
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    with deepspeed.zero.Init(
        data_parallel_group=mpu.get_data_parallel_group(),
        remote_device=None if args.remote_device == "none" else args.remote_device,
        config_dict_or_path=args.deepspeed_config,
        enabled=args.zero_stage == 3,
        mpu=mpu,
    ):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = CodeGeeXModelPipe(num_tokentypes=0, parallel_output=True)
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(
                torch.ones(
                    (1, args.seq_length, args.seq_length),
                    device=torch.cuda.current_device(),
                )
            ).view(1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

        else:
            model = CodeGeeXModel(
                num_tokentypes=0,
                parallel_output=True,
            )
            
            if args.load_state is not None:
                timers = get_timers()
                print_rank_0("Loading warmstarting model states ...")
                timers("load-model-states").start()
                mp_rank = mpu.get_tensor_model_parallel_rank()
                if os.path.isdir(args.load_state):
                    model_path = os.path.join(
                        args.load_state, "mp_rank_{:02d}_model_states.pt".format(mp_rank)
                    )
                else:
                    model_path = args.load_state
                print_rank_0(f"Loading model from {model_path} ...")
                state_dict = torch.load(model_path, map_location="cpu")
                if "module" in state_dict:
                    state_dict = state_dict["module"]  # strip other client states
                model.load_state_dict(state_dict)
                timers("load-model-states").stop()
                timers.log(["load-model-states"])
    see_memory_usage(f"After Building Model", force=True)
    
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["input_ids", "attention_mask", "labels"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["input_ids"].contiguous()
    # attn_mask_ = data_b["attention_mask"].contiguous()
    labels_ = data_b["labels"].contiguous()

    tokens = tokens_[:, :-1]
    labels = labels_[:, 1:]

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    # mask loss to avoid predicting prompt and paddings
    prompt_loss_mask = labels >= 0
    loss_mask = prompt_loss_mask * loss_mask

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["input_ids"]
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["input_ids"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )
    
    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, output_tensor):
    args = get_args()

    def compute_lm_loss(losses: torch.Tensor, loss_mask: torch.Tensor):
        if args.gold:
            losses_ = losses.detach()
            prob = torch.exp(-losses_)  # Pθ(s)
            torch.sqrt_(prob)  # Pθ(s)ᵃ
            torch.clamp_min_(prob, args.gold_beta)  # max(Pθ(s)ᵃ,β)
            losses = prob * losses

        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / torch.clamp_min(loss_mask.sum(), 1e-8)

        return loss

    losses = output_tensor.float()
    loss = compute_lm_loss(losses, loss_mask)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def valid_loss_func(loss_mask, output_tensor):
    args = get_args()

    def compute_lm_loss(losses: torch.Tensor, loss_mask: torch.Tensor):
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / torch.clamp_min(loss_mask.sum(), 1e-8)

        return loss

    losses = output_tensor.float()
    loss = compute_lm_loss(losses, loss_mask)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    
    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def valid_forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(valid_loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    
    print_rank_0("> building train, validation, and test datasets " "for GPT ...")
    if args.co_evaluation:
        def dataset_partition_path_parsing(data_path):
            dataset_path = {}
            for index in range(len(data_path)):
                dataset_path[data_path[index]] = data_path[index]
            return dataset_path
        assert args.valid_data_path is not None, "Valid data path must be given when --co-evaluation is turned on."
        valid_data_path = dataset_partition_path_parsing(args.valid_data_path)
        if args.test_data_path is not None:
            test_data_path = dataset_partition_path_parsing(args.test_data_path)
        else:
            test_data_path = None
        train_ds, _, _ = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string="1,0,0",
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
        )
        valid_ds = {}
        for key, value in valid_data_path.items():
            _, valid_ds_item, _ = build_train_valid_test_datasets(
                data_prefix=[value],
                data_impl=args.data_impl,
                splits_string="0,1,0",
                train_valid_test_num_samples=train_val_test_num_samples,
                seq_length=args.seq_length,
                seed=args.seed,
                skip_warmup=(not args.mmap_warmup),
            )
            valid_ds[key] = valid_ds_item
        if test_data_path is not None:
            test_ds = {}
            for key, value in test_data_path.items():
                _, _, test_ds_item = build_train_valid_test_datasets(
                    data_prefix=[value],
                    data_impl=args.data_impl,
                    splits_string="0,0,1",
                    train_valid_test_num_samples=train_val_test_num_samples,
                    seq_length=args.seq_length,
                    seed=args.seed,
                    skip_warmup=(not args.mmap_warmup),
                )
                test_ds[key] = test_ds_item
        else:
            test_ds = None
    elif args.valid_data_path is None:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
        )
    else:
        train_ds, _, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string="100,0,0",
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
        )

        _, valid_ds, _ = build_train_valid_test_datasets(
            data_prefix=args.valid_data_path,
            data_impl=args.data_impl,
            splits_string="0,100,0",
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
        )

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        valid_forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )