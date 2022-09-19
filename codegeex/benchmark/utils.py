import gzip
import json
import os
from typing import *

LANGUAGE_TAG = {
    "c++"          : "// language: C++",
    "cpp"          : "// language: C++",
    "c"            : "// language: C",
    "c#"           : "// language: C#",
    "cuda"         : "// language: Cuda",
    "objective-c"  : "// language: Objective-C",
    "objective-c++": "// language: Objective-C++",
    "python"       : "# language: Python",
    "java"         : "// language: Java",
    "scala"        : "// language: Scala",
    "tex"          : f"% language: TeX",
    "html"         : "<!--language: HTML-->",
    "php"          : "// language: PHP",
    "js"           : "// language: JavaScript",
    "javascript"   : "// language: JavaScript",
    "typescript"   : "// language: TypeScript",
    "go"           : "// language: Go",
    "shell"        : "# language: Shell",
    "rust"         : "// language: Rust",
    "css"          : "/* language: CSS */",
    "sql"          : "-- language: SQL",
    "kotlin"       : "// language: Kotlin",
    "pascal"       : "// language: Pascal",
    "r"            : "# language: R",
    "fortran"      : "!language: Fortran",
    "lean"         : "-- language: Lean",
}

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go"    : [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp"   : [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}


def read_dataset(
        data_file: str = None,
        dataset_type: str = "humaneval",
        split: str = "test",
        args=None,
        num_shot=None,
) -> Dict:
    if num_shot is not None:
        print(f"{num_shot}-shot setting...")
    if "humaneval" in dataset_type.lower():
        if data_file is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(current_path, "..", "humaneval-x", "python", "data", "humaneval_python.jsonl.gz")
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset


def read_translation_dataset(
        data_file_src: str = None,
        data_file_tgt: str = None,
        lang_src: str = None,
        lang_tgt: str = None,
        dataset_type: str = "humaneval",
) -> Dict:
    if "humaneval" in dataset_type.lower():
        dataset_src = {task["task_id"]: task for task in stream_jsonl(data_file_src)}
        dataset_tgt = {task["task_id"].split("/")[-1]: task for task in stream_jsonl(data_file_tgt)}
        for k, sample in dataset_src.items():
            prompt = "code translation\n"
            if lang_src == "cpp":
                prompt += "C++:\n"
            elif lang_src == "js":
                prompt += "JavaScript:\n"
            else:
                prompt += f"{lang_src}:\n".capitalize()
            prompt += dataset_src[k]["declaration"] + "\n" + dataset_src[k]["canonical_solution"].rstrip() + "\n"
            if lang_tgt == "cpp":
                prompt += "C++:\n"
            elif lang_tgt == "js":
                prompt += "JavaScript:\n"
            else:
                prompt += f"{lang_tgt}:\n".capitalize()
            prompt += dataset_tgt[k.split("/")[-1]]["declaration"]
            dataset_src[k]["prompt"] = prompt
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset_src


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def process_extra_prompt(prompt: str, language_type: str = None) -> str:
    """
    Processes the extra prompt.
    """
    language = language_type.lower()
    if language in LANGUAGE_TAG:
        extra_prompt = LANGUAGE_TAG[language] + "\n"
    else:
        extra_prompt = ""

    return extra_prompt + prompt


def is_code_generation_finished(
        code: str,
        language_type: str = None,
        dataset: str = None,
):
    """
    Checks whether the generated code is finished.
    """
    if language_type is None or dataset is None:
        return False

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            for line in code.split("\n"):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return True
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
            for w in end_words:
                if w in code:
                    return True
        elif language_type.lower() == "java":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "go":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "js":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "cpp":
            if code.count("{") + 1 == code.count("}"):
                return True

    return False


def is_code_generation_finished_fix(
        code: str,
        language_type: str = None,
        dataset: str = None,
):
    """
    Checks whether the generated code is finished.
    """
    if language_type is None or dataset is None:
        return False

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            for line in code.split("\n"):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return True
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
            for w in end_words:
                if w in code:
                    return True
        elif language_type.lower() == "java":
            if code.count("{") == code.count("}"):
                return True
        elif language_type.lower() == "go":
            if code.count("{") == code.count("}"):
                return True
        elif language_type.lower() == "js":
            if code.count("{") == code.count("}"):
                return True
        elif language_type.lower() == "cpp":
            if code.count("{") == code.count("}"):
                return True

    return False


def cleanup_code(
        code: str,
        language_type: str = None,
        dataset: str = None,
):
    """
    Cleans up the generated code.
    """
    if language_type is None or dataset is None:
        return code

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint", "\nassert"]
            for w in end_words:
                if w in code:
                    code = code[:code.rfind(w)]
        elif language_type.lower() == "java":
            main_pos = code.find("public static void main")
            if main_pos != -1:
                code = code[:main_pos] + '}'
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
            if code.count('{') + 1 == code.count('}'):
                code += "\n}"
        elif language_type.lower() == "go":
            end_words = ["\n//", "\nfunc main("]
            for w in end_words:
                if w in code:
                    code = code[:code.rfind(w)]
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "cpp":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "js":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'

    return code
