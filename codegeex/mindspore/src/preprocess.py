# Copyright 2020 Huawei Technologies Co., Ltd
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
transform wikitext-2, wikitext-103, lambada, openwebtext dataset to mindrecord.
"""
import argparse
import glob
import json
import os
import re
from functools import partial
from multiprocessing import Pool, current_process

import numpy as np
from mindspore.mindrecord import FileWriter
from src.tokenization_jieba import JIEBATokenizer


def chunks(lst, n):
    """yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def package_file(it, n):
    """package multiple files"""
    stop = False
    while not stop:
        batch = []
        for _ in range(n):
            try:
                batch.append(next(it))
            except StopIteration:
                stop = True
        if not batch:
            break
        yield batch


def clean_wikitext(string):
    """cleaning wikitext dataset"""
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def tokenize_openwebtext(tokenizer, iterator, seq_length, eot):
    """tokenize openwebtext dataset"""
    for file_path in iterator:
        if os.path.getsize(file_path) == 0:
            continue
        content = []
        with open(file_path, "r", encoding="utf-8") as f:
            for para in f.read().split("\n\n"):
                if para:
                    tokenized_text = tokenizer.tokenize(para)
                    content += tokenizer.convert_tokens_to_ids(
                        tokenized_text
                    ) + [eot]
        for chunk in chunks(content, seq_length):
            sample = {}
            if len(chunk) == seq_length:
                sample["input_ids"] = np.array(chunk, dtype=np.int32)
                yield sample


def tokenize_wiki(tokenizer, file_path, seq_length, eot):
    """tokenize wikitext-2/wikitext-103 dataset"""
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for para in clean_wikitext(f.read()).split("\n\n"):
            if para and para.strip().startswith("=") is False:
                tokenized_text = tokenizer.tokenize(para)
                content += tokenizer.convert_tokens_to_ids(tokenized_text) + [
                    eot
                ]
    for chunk in chunks(content, seq_length):
        sample = {}
        if len(chunk) == seq_length:
            sample["input_ids"] = np.array(chunk, dtype=np.int32)
            yield sample


def tokenize_lambada(tokenizer, file_path, seq_length, eot):
    """tokenize lambada dataset"""
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            para = (
                json.loads(line)["text"]
                .replace("“", '"')
                .replace("”", '"')
                .strip()
                .strip(".")
            )
            tokenized_text = tokenizer.tokenize(para)
            content += tokenizer.convert_tokens_to_ids(tokenized_text) + [eot]
    for chunk in chunks(content, seq_length):
        sample = {}
        if len(chunk) == seq_length:
            sample["input_ids"] = np.array(chunk, dtype=np.int32)
            yield sample


def task_unit(iterator, tokenizer, seq_length, eot, parallel_writer=True):
    """task for each process"""
    p = current_process()
    index = p.pid if p.pid else 0

    item_iter = tokenize_openwebtext(tokenizer, iterator, seq_length, eot)
    batch_size = 1024  # size of write batch
    count = 0
    while True:
        data_batch = []
        try:
            for _ in range(batch_size):
                data_batch.append(next(item_iter))
                count += 1
            writer.write_raw_data(data_batch, parallel_writer=parallel_writer)
            print("Process {} transformed {} records.".format(index, count))
        except StopIteration:
            if data_batch:
                writer.write_raw_data(
                    data_batch, parallel_writer=parallel_writer
                )
                print("Process {} transformed {} records.".format(index, count))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="openwebtext")
    parser.add_argument("--input_glob", type=str, default="*.txt")
    parser.add_argument(
        "--output_file", type=str, default="./output/transfered_mindrecord"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt", choices=["gpt", "jieba"]
    )
    parser.add_argument("--vocab_file", type=str, default=None)
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--file_partition", type=int, default=1)
    parser.add_argument("--file_batch_size", type=int, default=1024)
    parser.add_argument("--num_process", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=1025)
    parser.add_argument("--eot", type=int, default=50256)
    parser.add_argument("--data_column_name", type=str, default="input_ids")

    args = parser.parse_args()

    out_dir, out_file = os.path.split(os.path.abspath(args.output_file))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    schema = {
        args.data_column_name: {"type": "int32", "shape": [-1]},
    }
    writer = FileWriter(
        file_name=args.output_file, shard_num=args.file_partition
    )
    writer.add_schema(schema, args.dataset_type)
    writer.open_and_set_header()

    # Start to load tokenizer
    if args.tokenizer == "gpt":
        try:
            from transformers import GPT2Tokenizer
        except ModuleNotFoundError:
            print("module 'transformers' not installed.")
        word_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        if not os.path.exists(args.vocab_file):
            raise FileNotFoundError(f"file {args.vocab_file} do not exists.")
        if not os.path.exists(args.model_file):
            raise FileNotFoundError(f"file {args.model_file} do not exists.")
        word_tokenizer = JIEBATokenizer(
            vocab_file=args.vocab_file, model_file=args.model_file
        )

    transforms_count = 0
    if args.dataset_type == "wiki":
        for x in tokenize_wiki(
                word_tokenizer, args.input_glob, args.seq_length, args.eot
        ):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == "lambada":
        for x in tokenize_lambada(
                word_tokenizer, args.input_glob, args.seq_length, args.eot
        ):
            transforms_count += 1
            writer.write_raw_data([x])
        print("Transformed {} records.".format(transforms_count))
    elif args.dataset_type == "openwebtext":
        file_iter = glob.iglob(args.input_glob)
        with Pool(processes=args.num_process) as pool:
            map_func = partial(
                task_unit,
                tokenizer=word_tokenizer,
                seq_length=args.seq_length,
                eot=args.eot,
            )
            pool.map(map_func, package_file(file_iter, args.file_batch_size))
    else:
        raise ValueError(
            "Not support dataset type: {}".format(args.dataset_type)
        )

    writer.commit()
    out_file = args.output_file
    if args.file_partition > 1:
        out_file += "0"
    print("Transform finished, output files refer: {}".format(out_file))
