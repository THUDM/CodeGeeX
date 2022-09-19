# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Create dataset for training and evaluating
"""

import os
from copy import deepcopy

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import numpy as np

from .sat_dataset import (
    LMDBDataset,
    PadDataset,
    ConcatDataset,
    SubsetDataset,
)


def get_input_data_batch_slice_map(input_ids, eod_id, rank, dis, eod_reset):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset
    Inputs:
        input_ids: the input token ids
        eod_id: the id for <EOD>
        rank: the current rank
        dis: the slice value for each rank
        eod_reset: whether to open eod reset or not
    returns:
        input_ids: the input token ids
        position_id: the position ids cosidering eod reset
        attention_mask: the attention mask considering eod reset
    """
    # rank = int(rank)
    # input_ids = input_ids[rank * dis : (rank + 1) * dis]
    if np.any(input_ids > 60000):
        raise ValueError("==exceed error")
    # print("===input_ids tpye: ", input_ids.dtype, flush=True) 
    if not eod_reset:
        return input_ids
    seq_length = input_ids.shape[1] - 1
    # Initialize position_ids and attention_mask
    batch_input_ids = deepcopy(input_ids)
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))

    # Loop through batches
    for bs_i in range(len(input_ids)):
        # Get normal position_ids and attention_mask
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        # Find eod_of_document
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            # Reset position_ids and attention_mask considering EOD
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask


def create_dataset(batch_size, data_path, args_opt, device_num=1, rank=0, drop=True, full_batch=False,
                   data_start_index=0,
                   eod_reset=False, eod_id=50256, column_name='input_ids', epoch=1, num_samples=None,
                   train_and_eval=False, val_ratio=0):
    """
    Create dataset
    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder
        eod_reset: whether enable position reset and attention mask reset
        eod_id: the id for <EOD>
        column_name: the column name of the mindrecord file. Default is input_ids
        epoch: The repeat times of the dataset
    Returns:
        dataset_restore: the dataset for training or evaluating
    """
    ds.config.set_seed(1)
    # Control the size of data queue in the consideration of the memory
    ds.config.set_prefetch_size(1)

    if full_batch:
        # no need to slice from the inputs
        rank = 0
        dis = batch_size
    else:
        # Each card slice a small batch from the full batch
        dis = int(batch_size / device_num)
        if batch_size % device_num != 0:
            raise ValueError(
                f"batch size {batch_size} should be a multiple of device number {device_num}."
                " You should change the args: per_batch_size."
            )

    skip_num = args_opt.has_trained_steps * dis
    # skip_num = 0
    num_parallel_workers = 4
    train_data = get_code_data_train(data_path, args_opt, skip_num=(skip_num // num_parallel_workers))
    if train_and_eval:
        val_data = get_code_data_eval("/home/work/sfs/xx/data_valid",
                                      args_opt)  # TODO: set as current validation set path
    else:
        val_data = None

    dataset_train = ds.GeneratorDataset(train_data, column_names=[column_name], num_samples=num_samples,
                                        num_shards=device_num, shard_id=rank, shuffle=True,
                                        num_parallel_workers=num_parallel_workers)
    if train_and_eval:
        dataset_val = ds.GeneratorDataset(val_data, column_names=[column_name], num_samples=num_samples,
                                          num_shards=device_num, shard_id=rank, shuffle=True,
                                          num_parallel_workers=num_parallel_workers)
    else:
        dataset_val = None
    type_cast_op = C.TypeCast(mstype.int32)
    type_cast_op_float = C.TypeCast(mstype.float16)

    map_func = (lambda input_ids: get_input_data_batch_slice_map(input_ids, eod_id, rank, dis, eod_reset))
    # If eod_reset enabled, another two inputs will be generated through input_ids
    dataset_train = dataset_train.skip(skip_num)
    if eod_reset:
        dataset_train = dataset_train.batch(dis, drop_remainder=drop)
        dataset_train = dataset_train.map(operations=map_func, input_columns=[column_name],
                                          output_columns=[column_name, "position_id", "attention_mask"],
                                          column_order=[column_name, "position_id", "attention_mask"])
        dataset_train = dataset_train.map(input_columns="position_id", operations=type_cast_op)
        dataset_train = dataset_train.map(input_columns="attention_mask", operations=type_cast_op_float)
    else:
        dataset_train = dataset_train.map(input_columns=[column_name], operations=type_cast_op)
        dataset_train = dataset_train.batch(batch_size, drop_remainder=drop)
        dataset_train = dataset_train.map(operations=map_func, input_columns=[column_name],
                                          output_columns=[column_name])
    dataset_train = dataset_train.map(input_columns=column_name, operations=type_cast_op)
    dataset_train = dataset_train.repeat(epoch)

    if dataset_val is not None:
        if eod_reset:
            dataset_val = dataset_val.batch(dis, drop_remainder=drop)
            dataset_val = dataset_val.map(operations=map_func, input_columns=[column_name],
                                          output_columns=[column_name, "position_id", "attention_mask"],
                                          column_order=[column_name, "position_id", "attention_mask"])
            dataset_val = dataset_val.map(input_columns="position_id", operations=type_cast_op)
            dataset_val = dataset_val.map(input_columns="attention_mask", operations=type_cast_op_float)
        else:
            dataset_val = dataset_val.map(input_columns=[column_name], operations=type_cast_op)
            dataset_val = dataset_val.batch(batch_size, drop_remainder=drop)
            dataset_val = dataset_val.map(operations=map_func, input_columns=[column_name],
                                          output_columns=[column_name])
        dataset_val = dataset_val.map(input_columns=column_name, operations=type_cast_op)
    return dataset_train, dataset_val


def get_code_data_train(code_data_path, args_opt, process_fn=None, scale=1, skip_num=0):
    datasets = []
    paths = []

    for dir in sorted(os.listdir(code_data_path)):
        sub_dirs = os.listdir(os.path.join(code_data_path, dir))
        for sub_dir in sub_dirs:
            if os.path.exists(os.path.join(code_data_path, dir, sub_dir, 'data.mdb')) and os.path.exists(
                    os.path.join(code_data_path, dir, sub_dir, 'lock.mdb')):
                paths.append(os.path.join(code_data_path, dir, sub_dir))

    for full_path in paths:
        if os.path.isdir(full_path):
            print(f"Loading code data {full_path}")
            data = LMDBDataset(
                full_path,
                process_fn=process_fn,
            )
            data = PadDataset(
                data,
                args_opt.seq_length,
                args_opt.eod_id,
            )
            datasets.append(data)
    datasets = ConcatDataset(datasets, skip_num=skip_num)
    return datasets


def get_code_data_eval(code_data_path, args_opt, process_fn=None, scale=1):
    datasets = []
    paths = []

    for dir in sorted(os.listdir(code_data_path)):
        sub_dirs = os.listdir(os.path.join(code_data_path, dir))
        for sub_dir in sub_dirs:
            if os.path.exists(os.path.join(code_data_path, dir, sub_dir, 'data.mdb')) and os.path.exists(
                    os.path.join(code_data_path, dir, sub_dir, 'lock.mdb')):
                paths.append(os.path.join(code_data_path, dir, sub_dir))

    for full_path in paths:
        if os.path.isdir(full_path):
            print(f"Loading code data {full_path}")
            data = LMDBDataset(
                full_path,
                process_fn=process_fn,
            )
            data = PadDataset(
                data,
                args_opt.seq_length,
                args_opt.eod_id,
            )
            data = SubsetDataset(
                data,
                0,
                len(data) // 10,
            )
            datasets.append(data)
    datasets = ConcatDataset(datasets)
    print(f"==valid dataset has {len(datasets)} items")
    return datasets
