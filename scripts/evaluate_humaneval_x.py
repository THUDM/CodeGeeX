import argparse
import os
from pathlib import Path
from codegeex.benchmark.evaluate_humaneval_x import evaluate_functional_correctness
#GLOBALS
INPUT_FILE: str  
LANGUAGE: str  
N_WORKERS: int  
TIMEOUT: int 


parser = argparse.ArgumentParser("Debugging evaluate humaneval_x")
# Path to the .jsonl file that contains the generated codes.
parser.add_argument("-s","--samples", type=str)

# Target programming language, currently support one of ["python", "java", "cpp", "js", "go"]
parser.add_argument("-l","--language", default="python", type=str)

# Number of parallel workers.
parser.add_argument("-w","--workers", default=64, type=int)

# Timeout in seconds.
parser.add_argument("-t","--timeout", default=5, type=int)

args = parser.parse_args()

INPUT_FILE = args.samples
LANGUAGE = args.language  
N_WORKERS = args.workers  
TIMEOUT= args.timeout



SCRIPT_PATH: str = Path(os.path.abspath(__file__))
print(SCRIPT_PATH)
SCRIPT_DIR: str = os.path.dirname(SCRIPT_PATH)
print(SCRIPT_DIR)
MAIN_DIR: str = os.path.dirname(SCRIPT_DIR)
print(MAIN_DIR)

DATA_DIR=os.path.join(MAIN_DIR,"codegeex/benchmark/humaneval-x/" + LANGUAGE + "/data/humaneval_" + LANGUAGE + ".jsonl.gz")
print(DATA_DIR)

TMP_DIR=os.path.join(MAIN_DIR, "/codegeex/benchmark/humaneval-x/")


#Debugging
INPUT_FILE='/home/rog0d/Escritorio/CodeGeeX/generations/humaneval_rust_generations.jsonl.gz'
LANGUAGE='rust'
DATA_DIR=os.path.join(MAIN_DIR,"codegeex/benchmark/humaneval-x/" + LANGUAGE + "/data/humaneval_" + LANGUAGE + ".jsonl.gz")

"""
input_file: str = None,
        tmp_dir: str = "./",
        n_workers: int = 32,
        timeout: float = 5.0,
        problem_file: str = "../data/humaneval_python.jsonl.gz",
        out_dir: str = None,
        k: List[int] = [1, 10, 100],
        test_groundtruth: bool = False,
        example_test: bool = False,

"""

evaluate_functional_correctness(input_file=INPUT_FILE,
                                n_workers=N_WORKERS,
                                tmp_dir=TMP_DIR,
                                problem_file=DATA_DIR,
                                timeout=300.0)


