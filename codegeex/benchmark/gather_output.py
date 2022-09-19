import os
import sys
import fire
import glob


def gather_output(
    output_dir: str = "./output",
    output_prefix: str = None,
    if_remove_rank_files: int = 0,
):
    if output_prefix is None:
        output_list = glob.glob(output_dir + "/*")
    else:
        output_list = glob.glob(os.path.join(output_dir, output_prefix + "*"))

    for output_file in output_list:
        if "rank0" in output_file:
            output_prefix_ = output_file.split("_rank0.jsonl")[0]
            rank_files = glob.glob(output_prefix_ + "_rank*")
            with open(output_prefix_ + ".jsonl", "w") as f_out:
                for rank_file in rank_files:
                    with open(rank_file, "r") as f_in:
                        for line in f_in:
                            f_out.write(line)
                        if if_remove_rank_files:
                            os.remove(rank_file)
                            print(f"Removing {rank_file}...")

    if output_prefix is None:
        output_list = glob.glob(output_dir + "/*")
    else:
        output_list = glob.glob(os.path.join(output_dir, output_prefix + "*"))

    for output_file in output_list:
        if "rank" in output_file or "_unfinished" in output_file or "all" in output_file or "_result" in output_file:
            continue
        if "_finished" not in output_file:
            continue
        output_prefix_ = output_file.split("_finished.jsonl")[0]
        files = [output_file, output_prefix_ + "_unfinished.jsonl"]
        with open(output_prefix_ + "_all.jsonl", "w") as f_out:
            for f in files:
                with open(f, "r") as f_in:
                    for line in f_in:
                        f_out.write(line)

        print("Gathering finished. Saved in {}".format(output_prefix_ + "_all.jsonl"))


def main():
    fire.Fire(gather_output)


if __name__ == "__main__":
    sys.exit(main())
