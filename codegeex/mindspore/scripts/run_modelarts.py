import argparse
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, required=True)
parser.add_argument("--script", type=str, required=True)
parser.add_argument("--data_url", type=str, default=None)
parser.add_argument("--train_url", type=str, default=None)

args = parser.parse_args()

log_path = os.path.join(args.work_dir, "logs", os.environ.get("JOB_ID"), f'device{os.environ.get("RANK_ID")}')
tb_path = os.path.join(args.work_dir, "runs", os.environ.get("JOB_ID"))

Path(log_path).mkdir(parents=True, exist_ok=True)
Path(tb_path).mkdir(parents=True, exist_ok=True)

log_path_prefix_1 = os.path.join(args.work_dir, "logs")

os.environ["LOG_PATH"] = tb_path

print("=================RANK_TABLE_FILE: ", os.environ.get("RANK_TABLE_FILE", "not set"), flush=True)
print("=================ms import done", flush=True)
time.sleep(10)
# Use configurable temp directory (platform-specific for ModelArts)
temp_dir = os.environ.get("MODELARTS_TEMP_DIR", "/home/work/sfs/xx")
rank_table_source = "/home/work/rank_table/jobstart_hccl.json"
if os.path.exists(rank_table_source):
    os.system(f"cp {rank_table_source} {temp_dir}; sudo chmod +777 {rank_table_source}")
else:
    print(f"Warning: {rank_table_source} does not exist. Skipping copy.")
ret = os.system(f"cd {log_path} && bash {args.script} 2>&1 | tee output.log")
if os.environ.get("RANK_ID") == 0:
    log_dir = os.path.join(args.work_dir, "logs", os.environ.get("JOB_ID"))
    os.system(f"sudo chmod +777 -R {tb_path}")
    os.system(f"sudo chmod +777 -R {log_dir}")
print("==========ret code is: ", ret, flush=True)
if ret != 0:
    raise RuntimeError("ret code is :" + str(ret))
