import os
import zmq
import time
import torch
import random
import socket
import logging
import argparse

from typing import *
from codegeex.benchmark.utils import read_translation_dataset
from codegeex.megatron import get_args
from codegeex.megatron.inference import run_generation_distributed, model_provider
from codegeex.megatron.initialize import initialize_megatron

logging.getLogger("torch").setLevel(logging.WARNING)


def add_code_generate_args(parser):
    """Code generation arguments."""
    group = parser.add_argument_group(title="code generation")

    group.add_argument(
        "--hostfile",
        type=str,
        default="./hostfile",
    )
    group.add_argument(
        "--channel-ip",
        type=str,
        default=None,
        help="IP for ZeroMQ channel",
    )
    group.add_argument(
        "--channel-port",
        type=int,
        default=5555,
        help="Port for ZeroMQ channel",
    )
    group.add_argument(
        "--master-port",
        type=int,
        default=5666,
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    group.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy sampling.",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top p sampling.",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top k sampling.",
    )
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=1024,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--input-path",
        type=str,
        default="./benchmark/humaneval/HumanEval.jsonl",
        help="Get input path",
    )
    group.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to generate",
    )
    group.add_argument(
        "--recompute",
        action="store_true",
        help="During generation recompute all attention "
             "instead of using previously computed keys/values.",
    )
    group.add_argument(
        "--load-deepspeed",
        action="store_true",
        help="Load DeepSpeed checkpoint",
    )
    group.add_argument(
        "--ws-encoding-start-id",
        type=int,
        default=None,
        help="Start id for whitespace encoding",
    )
    group.add_argument(
        "--ws-encoding-length",
        type=int,
        default=None,
        help="Length of whitespace encoding",
    )
    group.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
    )
    group.add_argument(
        "--samples-per-problem",
        type=int,
        default=200,
        help="Number of samples to generate for each problem",
    )
    group.add_argument(
        "--output-prefix",
        type=str,
        default="./output/humaneval",
        help="Prefix for output files",
    )
    group.add_argument(
        "--gen-node-rank",
        type=int,
        default=None,
    )
    group.add_argument(
        "--gen-node-world-size",
        type=int,
        default=None,
    )
    group.add_argument(
        "--gen-world-size",
        type=int,
        default=1,
        help="Number of machines to use for generation",
    )
    group.add_argument(
        "--gen-rank",
        type=int,
        default=0,
        help="Machine rank for human eval generation",
    )
    group.add_argument(
        "--extra-prompt",
        type=str,
        default=None,
        help="Extra prompt to use for human eval generation",
    )
    group.add_argument(
        "--verbose-interval",
        type=int,
        default=100,
    )
    group.add_argument(
        "--problem-split",
        type=str,
        default="test",
    )
    group.add_argument(
        "--prompt-type",
        type=str,
        default="notag",
    )
    group.add_argument(
        "--num-devices-per-node",
        type=int,
        default=None,
    )
    group.add_argument(
        "--return-scores",
        action="store_true",
    )
    group.add_argument(
        "--free-guidance",
        action="store_true",
    )
    group.add_argument(
        "--guide-temp",
        type=float,
        default=1.5,
    )
    group.add_argument(
        "--attention-upweight",
        type=float,
        default=None,
    )
    group.add_argument(
        '--bad-ids',
        nargs="*",
        type=int,
        default=None,
        help='Identify the type of programming language to generate',
    )
    group.add_argument(
        "--src-path",
        type=str,
        default="",
        help="Get source path",
    )
    group.add_argument(
        "--tgt-path",
        type=str,
        default="",
        help="Get target path",
    )
    group.add_argument(
        '--language-src-type',
        type=str,
        default=None,
        help='Identify the type of programming language',
    )
    group.add_argument(
        '--language-tgt-type',
        type=str,
        default=None,
        help='Identify the type of programming language to translate',
    )

    return parser


def main(node_rank: int, local_rank: int, master_port: int, num_devices: int):
    """Main program."""
    os.environ["WORLD_SIZE"] = str(num_devices)
    os.environ["RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "0.0.0.0"
    os.environ["MASTER_PORT"] = f"{master_port}"

    initialize_megatron(
        extra_args_provider=add_code_generate_args,
        args_defaults={
            "tokenizer_type": "GPT2BPETokenizer",
            "no_load_rng"   : True,
            "no_load_optim" : True,
        },
    )

    # set_random_seed(node_rank * num_devices + local_rank)
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    world_size = args.gen_node_world_size * num_devices
    args.gen_rank = num_devices * node_rank + local_rank
    args.gen_world_size = world_size
    print(f"Generating on rank {args.gen_rank} of {args.gen_world_size}")

    # Set up model and load checkpoint.
    state_dict = torch.load(args.load, map_location="cpu")
    state_dict = state_dict["module"]

    print("Building CodeGeeX model ...")
    model = model_provider()
    model.load_state_dict(state_dict)
    model.eval()
    if args.fp16 and args.ln_fp16:
        model.half()
    model.cuda()

    # Generate samples.
    run_generation_distributed(model)

    print(f"(gen_rank={args.gen_rank}, rank={local_rank}) finished, waiting ...")
    torch.distributed.barrier()


def server():
    print(f"[ server ] starting ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel-ip",
        type=str,
        default=None,
        help="IP for ZeroMQ channel",
    )
    parser.add_argument(
        "--channel-port",
        type=int,
        default=5555,
        help="Port for ZeroMQ channel",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=6666,
        help="Port for torch distributed",
    )
    parser.add_argument(
        "--samples-per-problem",
        type=int,
        default=200,
        help="Number of samples to generate for each problem",
    )
    parser.add_argument(
        "--gen-node-world-size",
        type=int,
        default=1,
        help="Number of machines to use for generation",
    )
    parser.add_argument(
        "--src-path",
        type=str,
        default="",
        help="Get source path",
    )
    parser.add_argument(
        "--tgt-path",
        type=str,
        default="",
        help="Get target path",
    )
    parser.add_argument(
        "--problem-split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        '--language-src-type',
        type=str,
        default=None,
        help='Identify the type of programming language',
    )
    parser.add_argument(
        '--language-tgt-type',
        type=str,
        default=None,
        help='Identify the type of programming language to translate',
    )

    args = parser.parse_known_args()[0]

    entries = read_translation_dataset(args.src_path,
                                       args.tgt_path,
                                       lang_src=args.language_src_type,
                                       lang_tgt=args.language_tgt_type,
                                       dataset_type="humaneval")

    assert args.samples_per_problem % args.micro_batch_size == 0, "samples_per_problem should be divisible by micro_batch_size"

    res = []
    for entry in entries.values():
        res.extend([entry] * (args.samples_per_problem // args.micro_batch_size))
    random.shuffle(res)
    all_entries = res

    # setup zeromq channel
    print(f"[ server ] starting up on port {args.channel_port}", flush=True)
    context = zmq.Context()
    print(f"[ server ] creating socket", flush=True)
    socket = context.socket(zmq.REP)
    print(f"[ server ] binding to port {args.channel_port}", flush=True)
    socket.bind(f"tcp://*:{args.channel_port}")

    print(
        f"[ server ] loaded {len(entries)} entries, generated {len(all_entries)} samples",
        flush=True,
    )

    remaining_entries = all_entries.copy()
    running_workers = args.gen_node_world_size * torch.cuda.device_count()
    num_finished = 0

    print(f"[ server ] listening for requests ...", flush=True)
    start_time = time.perf_counter()
    while True:
        # Wait for next request from client
        msg = socket.recv_json()
        rank = msg["rank"]
        action = msg["action"]

        if action == "pull":
            if len(remaining_entries) == 0:
                print(f"[ server ] Shutting down worker {rank}", flush=True)
                socket.send_json({"task_id": None})
                running_workers -= 1
                if running_workers == 0:
                    print(f"[ server ] All workers finished", flush=True)
                    break
            else:
                entry = remaining_entries.pop()
                time_elapsed = time.perf_counter() - start_time
                print(f"[ server ] Sending entry {entry['task_id']} to worker {rank}", flush=True)
                remaining = (
                        len(remaining_entries)
                        / (len(all_entries) - len(remaining_entries))
                        * time_elapsed
                )
                time_per_sampple = 0.0 if num_finished == 0 else time_elapsed / num_finished / args.micro_batch_size
                print(
                    f"[ server ] total {len(all_entries)}, assigned {len(all_entries) - len(remaining_entries)}, "
                    f"finished {num_finished}, "
                    f"elapsed {time_elapsed:.4f}",
                    f"speed {time_per_sampple:.4f}s/sample",
                    f"remaining {remaining:.4f}",
                    flush=True,
                )
                socket.send_json({"task_id": entry})
        else:
            if action == "success":
                print(f"[ server ] {msg['task_id']} is finished", flush=True)
                socket.send_json({"pong": 1})
            else:
                print(f"[ server ] {msg['task_id']} is not finished", flush=True)
                remaining_entries.append(msg['task_id'])
                socket.send_json({"pong": 1})
                break

            num_finished += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostfile",
        type=str,
        default="./hostfile",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=5666,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_known_args()[0]

    print("start method: " + torch.multiprocessing.get_start_method())

    processes = []
    num_devices = torch.cuda.device_count()

    hosts = open(args.hostfile, "r").readlines()
    hosts = [host.strip() for host in hosts]
    master_port = args.master_port

    node_rank = None
    for i in range(len(hosts)):
        if hosts[i] == socket.gethostbyname(socket.gethostname()):
            node_rank = i
            break
    assert (
            node_rank is not None
    ), f"Could not find hostname ({socket.gethostbyname(socket.gethostname())}) in hostlist"

    # launch server
    if socket.gethostbyname(socket.gethostname()) == hosts[0]:
        server_process = torch.multiprocessing.Process(target=server)
        print(f"Launching server ...")
        server_process.start()
        processes.append(server_process)

    for i in range(num_devices):
        local_rank = i
        print(f"launching local rank {i}")

        p = torch.multiprocessing.Process(
            target=main,
            args=(node_rank, local_rank, master_port, num_devices),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
