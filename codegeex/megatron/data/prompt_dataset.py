# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT prompting dataset."""

import os
import time

import numpy as np
import torch

from codegeex.megatron import mpu, print_rank_0, get_tokenizer
from codegeex.megatron.data.blendable_dataset import BlendableDataset
from codegeex.megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from codegeex.megatron.data.dataset_utils import get_train_valid_test_split_
from codegeex.megatron.data.indexed_dataset import make_dataset as make_indexed_dataset


def build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
):
    """Build train, valid, and test datasets."""

    # Single dataset.
    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(
            data_prefix[0],
            data_impl,
            splits_string,
            train_valid_test_num_samples,
            seq_length,
            seed,
            skip_warmup,
        )

    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(
        data_prefix, train_valid_test_num_samples
    )
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i],
            data_impl,
            splits_string,
            datasets_train_valid_test_num_samples[i],
            seq_length,
            seed,
            skip_warmup,
        )
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    assert os.path.exists(data_prefix + "_input_ids.bin"), f"Input tokens datafile not found: {data_prefix}_input_ids.bin"
    assert os.path.exists(data_prefix + "_attention_mask.bin"), f"Attention mask datafile not found: {data_prefix}_attention_mask.bin"
    assert os.path.exists(data_prefix + "_labels.bin"), f"Labels datafile not found: {data_prefix}_labels.bin"

    input_ids_indexed_dataset = get_indexed_dataset_(data_prefix + "_input_ids", data_impl, skip_warmup)
    attention_mask_indexed_dataset = get_indexed_dataset_(data_prefix + "_attention_mask", data_impl, skip_warmup)
    labels_indexed_dataset = get_indexed_dataset_(data_prefix + "_labels", data_impl, skip_warmup)

    total_num_of_documents = input_ids_indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(
                start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32
            )
            dataset = PromptDataset(
                name,
                data_prefix,
                documents,
                input_ids_indexed_dataset,
                attention_mask_indexed_dataset,
                labels_indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
            )
        return dataset

    train_dataset = build_dataset(0, "train")
    valid_dataset = build_dataset(1, "valid")
    test_dataset = build_dataset(2, "test")
    print_rank_0(f"train_dataset:{type(train_dataset)}")
    print_rank_0(f"valid_dataset:{type(valid_dataset)}")
    print_rank_0(f"test_dataset:{type(test_dataset)}")

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    print_rank_0(
        " > finished creating indexed dataset in {:4f} "
        "seconds".format(time.time() - start_time)
    )
    print_rank_0("    number of documents: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class PromptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        documents,
        input_ids_indexed_dataset,
        attention_mask_index_dataset,
        labels_indexed_dataset,
        num_samples,
        seq_length,
        seed,
    ):
        """
        Args:
            name: name of the dataset.
            data_prefix: prefix of the data.
            documents: list of document indices.
            input_ids_indexed_dataset: indexed dataset for prompts.
            attention_mask_index_dataset: indexed dataset for text.
            labels_indexed_dataset: indexed dataset for labels.
            num_samples: number of samples to draw from the indexed dataset.
            seq_length: sequence length.
            seed: seed for random number generator.
        """

        self.name = name
        self.input_ids_indexed_dataset = input_ids_indexed_dataset
        self.attention_mask_index_dataset = attention_mask_index_dataset
        self.labels_indexed_dataset = labels_indexed_dataset
        self.seq_length = seq_length
        self.eod_token = get_tokenizer().eod

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < input_ids_indexed_dataset.sizes.shape[0]
        assert input_ids_indexed_dataset.sizes.shape[0] == attention_mask_index_dataset.sizes.shape[0]
        assert attention_mask_index_dataset.sizes.shape[0] == labels_indexed_dataset.sizes.shape[0]

        # Build index mappings.
        self.doc_idx = _build_index_mappings(
            self.name,
            data_prefix,
            documents,
            self.input_ids_indexed_dataset.sizes,
            num_samples,
            seq_length,
            seed,
        )

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.doc_idx.shape[0]

    def __getitem__(self, idx):
        # get the doc index
        doc_idx = self.doc_idx[idx]
        doc_idx = int(doc_idx)  # NumPy int => Python int

        input_ids = self.input_ids_indexed_dataset[doc_idx]
        # print_rank_0(f"input_ids={input_ids}")
        attention_mask = self.attention_mask_index_dataset[doc_idx]
        labels = self.labels_indexed_dataset[doc_idx]

        res = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
        }

        return res


def _build_index_mappings(
    name, data_prefix, documents, sizes, num_samples, seq_length, seed,
):
    """Build index mappings.
    We only have to build doc-idx in prompt dataset.

    Args:
        name: name of the dataset.
        data_prefix: prefix of the data.
        documents: list of document indices.
        sizes: sizes of the indexed dataset.
        num_samples: number of samples to draw from the indexed dataset.
        seq_length: sequence length.
        seed: seed for random number generator.
    """
    num_epochs = _num_epochs(documents.shape[0], num_samples)
    np_rng = np.random.RandomState(seed=seed)

    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"

    if torch.distributed.get_rank() == 0:
        if not os.path.isfile(doc_idx_filename):
            print_rank_0(
                " > WARNING: could not find index map files, building "
                "the indices on rank 0 ..."
            )

            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng, False)[:num_samples]
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)

            print_rank_0(
                " > elasped time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
            torch.distributed.get_world_size()
            // torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group())
    )
    # Load mappings.
    start_time = time.time()
    print_rank_0(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0("    total number of samples: {}".format(doc_idx.shape[0]))
    print_rank_0("    total number of epochs: {}".format(num_epochs))

    return doc_idx


def _num_epochs(samples_per_epoch, num_samples):
    """Calculate the epoch needed for so many sample."""
    return int(np.ceil(num_samples / samples_per_epoch))


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs - 1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))