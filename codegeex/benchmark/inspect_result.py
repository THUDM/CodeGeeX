import os
import sys
import fire
import json
import glob
import numpy as np
import pandas as pd

from collections import defaultdict
from codegeex.benchmark.metric import estimate_pass_at_k

error_types = {
    "python"    : [
        "accepted",
        "assertion error",
        "undefined error",
        "runtime error",
        "syntax error",
        "timeout error",
        "type error",
    ],
    "java"      : [
        "accepted",
        "compilation error",
        "assertion error",
        "timeout error",
        "index error",
        "class cast error",
        "stack overflow",
        "null pointer error",
        "unsupported operation",
        "number format error",
        "no such element",
        "illegal argument",
        "out of memory",
        "arithmetic error",
        "others",
    ],
    "cpp"       : [
        "accepted",
        "compilation error",
        "assertion error",
        "range error",
        "invalid argument",
        "pointer error",
        "out of memory",
        "package error",
        "others",
    ],
    "javascript": [
        "accepted",
        "assertion error",
        "undefined error",
        "runtime error",
        "syntax error",
        "timeout error",
        "range error",
        "type error",
    ],
    "go"        : [
        "accepted",
        "assertion error",
        "undefined error",
        "runtime error",
        "syntax error",
        "timeout error",
        "type error",
        "notused error",
    ],
}


def inspect_result(
        input_dir: str = None,
        input_file: str = None,
        output_dir: str = None,
        pass_at_k_outpath: str = None,
):
    if input_dir is not None:
        input_files = glob.glob(input_dir + "/*_results.jsonl")
    else:
        input_files = [input_file]

    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = "/"

    pass_at_k_outs = []
    for input_file in input_files:
        result_stats = defaultdict(dict)
        df = None
        incompleted = False
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                task_id = obj["task_id"]
                language_type = task_id.split("/")[0].lower()
                if language_type not in error_types:
                    if language_type == "humaneval":
                        language_type = "python"
                    elif language_type == "C++":
                        language_type = "cpp"
                    else:
                        incompleted = True
                        break
                if task_id not in result_stats.keys():
                    default_stats = {}
                    default_stats["task_id"] = task_id
                    default_stats["n_sample"] = 0
                    for k in error_types[language_type]:
                        default_stats[k] = 0

                    result_stats[task_id] = default_stats.copy()
                if df is None:
                    df = pd.DataFrame(columns=error_types[language_type])
                result_stats[task_id]["n_sample"] += 1

                if "failed" in obj["result"]:
                    error = obj["result"]
                    if language_type == "python":
                        if "assertionerror" in error.lower():
                            result_stats[task_id]["assertion error"] += 1
                        elif "syntax" in error.lower() or "indent" in error.lower() or "literal" in error.lower():
                            result_stats[task_id]["syntax error"] += 1
                        elif "not defined" in error.lower():
                            result_stats[task_id]["undefined error"] += 1
                        elif "timeout" in error.lower():
                            result_stats[task_id]["timeout error"] += 1
                        elif "type" in error.lower():
                            result_stats[task_id]["type error"] += 1
                        else:
                            result_stats[task_id]["runtime error"] += 1

                    elif language_type == "java":
                        if "wrong answer" in error:
                            result_stats[task_id]["assertion error"] += 1
                        elif "compilation error" in error:
                            result_stats[task_id]["compilation error"] += 1
                        elif "time out" in error:
                            result_stats[task_id]["timeout error"] += 1
                        elif "IndexOutOfBounds" in error:
                            result_stats[task_id]["index error"] += 1
                        elif "UnsupportedOperation" in error:
                            result_stats[task_id]["unsupported operation"] += 1
                        elif "ClassCast" in error:
                            result_stats[task_id]["class cast error"] += 1
                        elif "NullPointer" in error:
                            result_stats[task_id]["null pointer error"] += 1
                        elif "NumberFormat" in error:
                            result_stats[task_id]["number format error"] += 1
                        elif "NoSuchElement" in error:
                            result_stats[task_id]["no such element"] += 1
                        elif "StackOverflow" in error:
                            result_stats[task_id]["stack overflow"] += 1
                        elif "Arithmetic" in error:
                            result_stats[task_id]["arithmetic error"] += 1
                        elif "OutOfMemory" in error:
                            result_stats[task_id]["out of memory"] += 1
                        elif "IllegalArgument" in error:
                            result_stats[task_id]["illegal argument"] += 1
                        else:
                            result_stats[task_id]["others"] += 1

                    elif language_type == "cpp":
                        if "compilation error" in error.lower():
                            result_stats[task_id]["compilation error"] += 1
                        elif "int main(): assertion" in error.lower():
                            result_stats[task_id]["assertion error"] += 1
                        elif "out_of_range" in error.lower():
                            result_stats[task_id]['range error'] += 1
                        elif "corrupted top size" in error.lower():
                            result_stats[task_id]['range error'] += 1
                        elif "length_error" in error.lower():
                            result_stats[task_id]['range error'] += 1
                        elif "invalid_argument" in error.lower():
                            result_stats[task_id]['invalid argument'] += 1
                        elif "invalid pointer" in error.lower():
                            result_stats[task_id]['pointer error'] += 1
                        elif "double free" in error.lower():
                            result_stats[task_id]['pointer error'] += 1
                        elif "free()" in error.lower():
                            result_stats[task_id]['pointer error'] += 1
                        elif "logic_error" in error.lower():
                            result_stats[task_id]['pointer error'] += 1
                        elif "sysmalloc: assertion" in error.lower():
                            result_stats[task_id]['pointer error'] += 1
                        elif "stack smashing" in error.lower():
                            result_stats[task_id]['out of memory'] += 1
                        elif "bad_alloc" in error.lower():
                            result_stats[task_id]['out of memory'] += 1
                        elif "terminate called after throwing an instance of" in error.lower():
                            result_stats[task_id]['package error'] += 1
                        else:
                            result_stats[task_id]["others"] += 1

                    elif language_type == "javascript":
                        if "Assertion failed" in error:
                            result_stats[task_id]["assertion error"] += 1
                        elif "SyntaxError" in error:
                            result_stats[task_id]["syntax error"] += 1
                        elif "ReferenceError" in error:
                            result_stats[task_id]["undefined error"] += 1
                        elif "timed out" in error:
                            result_stats[task_id]["timeout error"] += 1
                        elif "TypeError" in error:
                            result_stats[task_id]["type error"] += 1
                        elif "RangeError" in error:
                            result_stats[task_id]["range error"] += 1
                        else:
                            result_stats[task_id]["runtime error"] += 1

                    elif language_type == "go":
                        if "Error:      \tNot equal:" in error:
                            result_stats[task_id]["assertion error"] += 1
                        elif "undefined" in error:
                            result_stats[task_id]["undefined error"] += 1
                        elif "expected" in error and "found" in error:
                            result_stats[task_id]["syntax error"] += 1
                        elif "illegal" in error:
                            result_stats[task_id]["syntax error"] += 1
                        elif "unexpected" in error:
                            result_stats[task_id]["syntax error"] += 1
                        elif "FAIL" in error:
                            result_stats[task_id]["runtime error"] += 1
                        elif "timed out" in error:
                            result_stats[task_id]["timeout error"] += 1
                        elif "not used" in error:
                            result_stats[task_id]['notused error'] += 1
                        elif "type" in error:
                            result_stats[task_id]['type error'] += 1
                    else:
                        incompleted = True
                        break
                else:
                    if obj["passed"]:
                        result_stats[task_id]["accepted"] += 1

        if incompleted:
            print(f"Language not supported, aborted. {input_file}")
        else:
            try:
                total, correct = [], []
                for k, res in result_stats.items():
                    total.append(res["n_sample"])
                    correct.append(res["accepted"])
                    df_res = pd.DataFrame(res, index=[int(k.split("/")[-1])])
                    df = pd.concat([df, df_res], axis=0)

                total = np.array(total)
                correct = np.array(correct)

                ks = [1, 10, 100, 1000]
                pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                            for k in ks if (total >= k).all()}

                print(pass_at_k)
                pass_at_k["file"] = input_file
                pass_at_k["n"] = res["n_sample"]
                pass_at_k_outs.append(pass_at_k)

                output_prefix = input_file.split("/")[-1].split(".jsonl")[0]
                output_file = os.path.join(output_dir, output_prefix + "_stats.xlsx")
                df = df.sort_index(ascending=True)
                df.to_excel(output_file)

                print(f"Stats saved in {output_file}")
            except Exception as e:
                print(e)
                print(f"Data incompleted, aborted. {input_file}")
                
    if pass_at_k_outpath is not None:
        jsonl_path = os.path.join(output_dir, pass_at_k_outpath)
        with open(jsonl_path, "w") as f_out:
            for p in pass_at_k_outs:
                f_out.write(json.dumps(p) + "\n")
        print(f"Pass at k saved in {jsonl_path}")


def main():
    fire.Fire(inspect_result)


if __name__ == "__main__":
    sys.exit(main())
