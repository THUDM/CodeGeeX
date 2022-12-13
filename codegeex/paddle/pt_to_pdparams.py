import argparse
import paddle
import torch

linear_layer = [
    "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h",
    "attention.query",
    "attention.key",
    "attention.value",
    "attention.dense",
]


def WalkDict(x):
    for i in x:
        if isinstance(x[i], dict):
            WalkDict(x[i])
        elif isinstance(x[i], torch.Tensor):
            print(f"Converting '{i}' from 'torch.Tensor' to 'numpy.ndarray'.")
            npy = x[i].cpu().numpy()
            if any([f".{layer}.weight" in i for layer in linear_layer]):
                print(f"Transposing linear layer weight '{i}'.")
                x[i] = npy.T
            else:
                x[i] = npy


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt",
        type=str,
        required=True,
        help="Path to pt checkpoint."
    )
    parser.add_argument(
        "--pdparams",
        type=str,
        required=True,
        help="Path to pdparams checkpoint."
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    state_dict = torch.load(opt.pt)
    WalkDict(state_dict)
    paddle.save(state_dict, opt.pdparams)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
