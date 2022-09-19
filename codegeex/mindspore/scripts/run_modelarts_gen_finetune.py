import argparse
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, required=True)
parser.add_argument("--script", type=str, required=True)
parser.add_argument("--data_url", type=str, default=None)
parser.add_argument("--train_url", type=str, default=None)
parser.add_argument("--language", type=str, default=None)

args = parser.parse_args()

log_path = os.path.join(args.work_dir, "logs", os.environ.get("JOB_ID"), f'device{os.environ.get("RANK_ID")}')
tb_path = os.path.join(args.work_dir, "runs", os.environ.get("JOB_ID"))

Path(log_path).mkdir(parents=True, exist_ok=True)
Path(tb_path).mkdir(parents=True, exist_ok=True)

log_path_prefix_1 = os.path.join(args.work_dir, "logs")

os.environ["LOG_PATH"] = tb_path
if args.language is not None:
    os.environ["LANGUAGE"] = args.language
else:
    os.environ["LANGUAGE"] = "Null"

print("=================RANK_TABLE_FILE: ", os.environ["RANK_TABLE_FILE"], flush=True)
print("=================ms import done", flush=True)
time.sleep(10)
os.system(
    "cp /home/work/rank_table/jobstart_hccl.json /home/work/sfs/xx; sudo chmod +777 /home/work/rank_table/jobstart_hccl.json")
ret = os.system(f"cd {log_path} && bash {args.script} 2>&1 | tee output.log")
if os.environ.get("RANK_ID") == 0:
    log_dir = os.path.join(args.work_dir, "logs", os.environ.get("JOB_ID"))
    os.system(f"sudo chmod +777 -R {tb_path}")
    os.system(f"sudo chmod +777 -R {log_dir}")
print("==========ret code is: ", ret, flush=True)
if ret != 0:
    raise RuntimeError("ret code is :" + str(ret))
