import os


ENV_NAMES = ["CUDA_HOME", "LD_LIBRARY_PATH", "PATH", "TORCH_EXTENSIONS_DIR", "CUDA_LAUNCH_BLOCKING"]


def main():
    s = ""
    for name in ENV_NAMES:
        if name in os.environ:
            value = os.environ[name]
            s += "{}={}\n".format(name, value)
            print(f"{name}={value}")
        else:
            print(f"{name} is not set")

    # write env vars to .deepspeed_env
    with open(".deepspeed_env", "w") as f:
        f.write(s)


if __name__ == "__main__":
    main()
