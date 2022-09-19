import pickle
from abc import ABC, abstractmethod
from bisect import bisect_right

import numpy as np


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class LMDBDataset(Dataset):
    def __init__(self, path, process_fn=None):
        import lmdb

        self.path = path
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.process_fn = process_fn
        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(f"Get {self.path}: {idx}")
        with self.env.begin(write=False) as txn:
            key = str(idx).encode("utf-8")
            # row = pickle.loads(txn.get(key))
            try:
                row = pickle.loads(txn.get(key))
            except TypeError:
                raise IndexError("Index out of range")
            if self.process_fn:
                return self.process_fn(row)
            else:
                return row


class PadDataset(Dataset):
    def __init__(self, dataset, seq_len, eod_id):
        self.dataset = dataset
        self.seq_len = seq_len + 1
        self.eod_id = eod_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx][0]
        return (item[:self.seq_len],) if self.seq_len <= len(item) else (
        np.concatenate((item, np.ones(self.seq_len - len(item)) * self.eod_id), axis=0),)
        # return (np.pad(item, (0, 1), constant_values=self.eod_id),)


class BinaryDataset(Dataset):
    def __init__(
            self,
            path,
            process_fn,
            length_per_sample=64 + 1024 + 4096,
            dtype="int32",
            preload=False,
            **kwargs,
    ):  # TODO ARGS
        assert length_per_sample is not None
        self.length_per_sample = length_per_sample
        self.dtype = np.dtype(dtype)
        self.process_fn = process_fn
        if preload:
            self.bin = np.fromfile(path, dtype=self.dtype).reshape(
                -1, length_per_sample
            )
        else:
            with open(path, "r") as fid:
                nbytes = fid.seek(0, 2)
                flen = fid.tell() // self.dtype.itemsize
            self.bin = np.memmap(
                path,
                dtype=self.dtype,
                shape=(flen // length_per_sample, length_per_sample),
            )

    def __len__(self):
        return self.bin.shape[0]

    def __getitem__(self, index):
        print(f"Get text: {index}")
        return self.process_fn(self.bin[index], index)


class TSVDataset(Dataset):
    def __init__(self, path, process_fn, with_heads=True, **kwargs):
        self.process_fn = process_fn
        with open(path, "r") as fin:
            if with_heads:
                self.heads = fin.readline().split("\t")
            else:
                self.heads = None
            self.items = [line.split("\t") for line in fin]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.process_fn(self.items[index])


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence, weights):
        r, s = [], 0
        for i, e in enumerate(sequence):
            l = int(len(e) * weights[i])
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, weights=None, skip_num=0, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = datasets
        if weights is None:
            self.weights = [1] * len(self.datasets)
        else:
            self.weights = weights
        self.cumulative_sizes = self.cumsum(self.datasets, self.weights)
        self.skip_num = skip_num
        self.visited_times = 0
        sample = self.datasets[0][0]
        self.echo = tuple(np.zeros_like(col).astype(np.int64) for col in sample)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if (self.visited_times % len(self)) < self.skip_num:
            self.visited_times += 1
            # print("===return zero", flush=True)
            return self.echo
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        return tuple(col.astype(np.int64) for col in self.datasets[dataset_idx][sample_idx])


class RandomMappingDataset(Dataset):
    """
    Dataset wrapper to randomly mapping indices to original order.
    Will also enlarge the length
    """

    def __init__(self, ds):
        self.wrapped_data = ds
        self.index_mapping = np.random.permutation(np.arange(len(ds)))

    def __len__(self):
        return len(self.wrapped_data)

    def __getitem__(self, index):
        # rng = random.Random(index)
        # rng = np.random.RandomState(
        #     seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)]
        # )
        # index = rng.randint(len(self.wrapped_data))
        return self.wrapped_data[self.index_mapping[index]]


class BlockedSplitDataset(Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Use block algorithm to reduce memory.
    In each block, using the `indices` items.
    """

    def __init__(self, ds, indices, block_size):
        if type(indices) is not np.ndarray:
            indices = np.array(indices)
        indices = np.sort(indices)
        self.block_size = block_size
        self.wrapped_data = ds
        self.wrapped_data_len = len(ds)
        self.indices = indices
        self.len = len(indices) * (len(ds) // block_size) + np.sum(indices < (len(ds) % block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.wrapped_data[
            (index // len(self.indices)) * self.block_size + self.indices[index % len(self.indices)]
            ]


class SubsetDataset(Dataset):
    def __init__(self, ds, start, length):
        assert start >= 0 and length > 0 and start + length <= len(ds), "Illegal start or length"
        self.ds = ds
        self.start = start
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")
        return self.ds[idx + self.start]


def split_train_val_test(ds, split=[0.99, 0.01, 0.0], seed=None):
    if seed is not None:
        np.random.seed(seed)
    if sum(split) == 0:
        raise Exception("Split cannot sum to 0.")
    if sum(split) > 1:
        raise Exception("Split portion exceeds 1.")
    start_idx = 0
    rtn_ds = [None] * len(split)
    proportions = [s * len(ds) for s in split]
    if sum(proportions) > len(ds):
        raise NotImplementedError("Split ratio exceeds in split_train_val_test!")
    random_mapping_ds = RandomMappingDataset(ds)
    for i, f in enumerate(split):
        proportion = int(len(random_mapping_ds) * f)
        if proportion != 0:
            rtn_ds[i] = SubsetDataset(ds=ds, start=start_idx, length=proportion)
            start_idx += proportion
    return rtn_ds

# def split_ds(ds, split=[0.99, 0.01, 0.0], seed=1):
#     """
#     Split a dataset into subsets given proportions of how
#     much to allocate per split. If a split is 0% returns None for that split.
#     Purpose: Useful for creating train/val/test splits
#     Arguments:
#         ds (Dataset or array-like): Data to be split.
#         split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
#         shuffle (boolean): Randomly split dataset. Default: True
#     """
#     np.random.seed(seed)
#     split_sum = sum(split)
#     if split_sum == 0:
#         raise Exception("Split cannot sum to 0.")
#     split = np.array(split, dtype=np.float32)
#     split /= split.sum()

#     start_idx = 0
#     residual_idx = 0
#     rtn_ds = [None] * len(split)
#     random_mapping_ds = RandomMappingDataset(ds)
#     for i, f in enumerate(split):
#         if f != 0:
#             proportion = len(random_mapping_ds) * split[i]


#     indices = np.random.permutation(np.array(range(block_size)))
#     for i, f in enumerate(split):
#         if f != 0:
#             proportion = block_size * split[i]
#             residual_idx += proportion % 1
#             split_ = int(int(proportion) + residual_idx)
#             rtn_ds[i] = BlockedSplitDataset(
#                 ds,
#                 indices[range(start_idx, start_idx + max(split_, 1))],
#                 block_size,
#             )
#             start_idx += split_
#             residual_idx %= 1
#     return rtn_ds
