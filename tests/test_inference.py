import os
import copy
import time
import torch
import random
import numpy as np

from codegeex.megatron import get_tokenizer, get_args
from codegeex.megatron.initialize import initialize_megatron
from codegeex.megatron.model import CodeGeeXModel
from codegeex.megatron.code_generation_utils import get_token_stream

torch.set_printoptions(precision=8)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_provider():
    """Build the model."""

    model = CodeGeeXModel(num_tokentypes=0,
                          parallel_output=False)

    return model


def add_code_generation_args(parser):
    """Code generation arguments."""
    group = parser.add_argument_group(title="code generation")

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
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--recompute",
        action="store_true",
        help="During generation recompute all attention "
             "instead of using previously computed keys/values.",
    )
    group.add_argument(
        "--ws-encoding-start-id",
        type=int,
        default=10,
        help="Start id for whitespace encoding",
    )
    group.add_argument(
        "--ws-encoding-length",
        type=int,
        default=80,
        help="Length of whitespace encoding",
    )
    group.add_argument(
        "--n-generation",
        type=int,
        default=10,
    )
    group.add_argument(
        "--eos-id",
        type=int,
        default=50256,
    )
    group.add_argument(
        "--prompt-file",
        type=str,
        default="./test_prompt.txt",
    )
    group.add_argument(
        "--perf-file",
        type=str,
        default="./perf_out.txt",
    )
    group.add_argument(
        "--perf-trace",
        type=str,
        default="./perf_out.txt",
    )
    group.add_argument(
        "--use-torch-profile",
        action="store_true",
    )
    group.add_argument(
        "--ln-fp32",
        action="store_true",
    )
    group.add_argument(
        '--bad-ids',
        nargs="*",
        type=int,
        default=None,
        help='Identify the type of programming language to generate',
    )

    return parser


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(random.randint(10000, 20000))

    initialize_megatron(
        extra_args_provider=add_code_generation_args,
    )

    args = get_args()
    set_random_seed(args.seed)

    print("Loading tokenizer ...")
    tokenizer = get_tokenizer()

    print("Loading state dict ...")
    state_dict = torch.load(args.load, map_location="cpu")
    state_dict = state_dict["module"]

    print("Building CodeGeeX model ...")
    model = model_provider()
    model.load_state_dict(state_dict)
    model.eval()
    if args.fp16 and args.ln_fp16:
        model.half()
    model.cuda()

    with open(args.prompt_file, "r") as f:
        prompt = f.readlines()
        prompt = "".join(prompt)

    print("Generating ...")
    t0 = time.perf_counter()
    for prompt in [prompt]:
        tokens = tokenizer.tokenize(prompt)
        print(tokens)
        print("Current prompt:")
        print(prompt)
        n_token_prompt = len(tokens)
        print("N_token_prompt:", n_token_prompt)
        token_stream = get_token_stream(
            model,
            [copy.deepcopy(tokens) for _ in range(args.micro_batch_size)],
            micro_batch_size=args.micro_batch_size,
            bad_ids=args.bad_ids,
        )
        is_finished = [False for _ in range(args.micro_batch_size)]
        for i, generated in enumerate(token_stream):
            generated_tokens = generated[0]
            for j in range(args.micro_batch_size):
                if is_finished[j]:
                    continue
                if generated_tokens[j].cpu().numpy()[-1] == tokenizer.eod or len(
                        generated_tokens[j]) >= args.out_seq_length:
                    is_finished[j] = True
                    generated_tokens_ = generated_tokens[j].cpu().numpy().tolist()
                    generated_code = tokenizer.detokenize(generated_tokens_[n_token_prompt:])
                    t1 = time.perf_counter()
                    print("Total generation time:", t1 - t0, "# Tokens:", len(generated_tokens_) - n_token_prompt)
                    print(f"{(t1 - t0) / (len(generated_tokens_) - n_token_prompt)}s/token")
                    print("================================= Generated code:")
                    print(generated_code)
                    t0 = time.perf_counter()
                if all(is_finished):
                    break

    print("Generation finished.")


if __name__ == "__main__":
    main()
