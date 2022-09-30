# HumanEval-X: A new benchmark for Multilingual Program Synthesis

🌐 <a href="README_zh.md" target="_blank">中文</a>

HumanEval-X is a new benchmark for better evaluating the multilingual ability of code generation models. While previous works evaluate multilingual program synthesis under semantic similarity (e.g., [CodeBLEU](https://arxiv.org/abs/2009.10297)) which is often misleading, HumanEval-X evaluates the functional correctness of the generated programs. HumanEval-X consists of 820 high-quality human-crafted data samples (each with test cases) in Python, C++, Java, JavaScript, and Go, and can be used for various tasks.

<img src="../../resources/en/hx_tasks.png">

<p align="center"><i>An illustration of tasks supported by <b>HumanEval-X</b>. Declarations, docstrings, and solutions are marked with red, green, and blue respectively. <b>Code generation</b> uses declaration and docstring as input, to generate solution. <b>Code translation</b> uses declaration in both languages and translate the solution in source language to the one in target language.</i></p>

In HumanEval-X, every sample in each language contains declaration, docstring, and solution, which can be combined in various ways to support different downstream tasks including generation, translation, summarization, etc. We currently focus on two tasks: **code generation** and **code translation**. For code generation, the model uses declaration and docstring as input to generate the solution. For code translation, the model uses declarations in both languages and the solution in the source language as input, to generate solutions in the target language. We remove the description during code translation to prevent the model from directly solving the problem. For both tasks, we use the unbiased pass@k metric proposed in [Codex](https://arxiv.org/abs/2107.03374): $\text{pass}@k:= \mathbb{E}[1-\frac{\tbinom{n-c}{k}}{\tbinom{n}{k}}]$, with $n=200$ and $k\in(1,10,100)$.

## How to use HumanEval-X

Data are stored in ``codegeex/benchmark/humaneval-x/[LANG]/data/humaneval_[LANG].jsonl.gz``, using JSON list format. There are six keys:

*   ``task_id``: indicates the target language and ID of the problem. Language is one of ["Python", "Java", "JavaScript", "CPP", "Go"].
*   ``prompt``: the function declaration and docstring, used for code generation.
*   ``declaration``: only the function declaration, used for code translation. 
*   ``canonical_solution``: human-crafted example solutions.
*   ``test``: hidden test samples, used for evaluation.
*   ``example_test``: public test samples (appeared in prompt), used for evaluation. 

### Evaluation Environment

The evaluation of the generated codes involves compiling and running in multiple programming languages. The versions of the programming language environments and packages we use are as follows:

| Dependency | Version  |
| ---------- | -------- |
| Python     | 3.8.12   |
| JDK        | 18.0.2.1 |
| Node.js    | 16.14.0  |
| js-md5     | 0.7.3    |
| C++        | 11       |
| g++        | 7.5.0    |
| Boost      | 1.71.0   |
| OpenSSL    | 3.0.0    |
| go         | 1.18.4   |

In order to save everyone the trouble of setting up the environments for these languages, we build a Docker image with the required environments and CodeGeeX installed.

You can directly pull the image from Docker Hub:

```bash
docker pull rishubi/codegeex:latest
```

Alternatively, if you are familiar with Dockerfile, you can build the image from `codegeex/docker/Dockerfile` or configure the Dockerfile as you like it:

```bash
cd codegeex/docker
docker build [OPTIONS] .
```

After obtaining the image, you can build a container using the following command:

```bash
docker run -it --gpus all --mount type=bind,source=<LOCAL PATH>,target=<PATH IN CONTAINER> [OPTIONS] <IMAGE NAME:TAG>
```

### Evaluation

We recommend evaluating in [the provided image](#evaluation-environment). To evaluate the generated samples, save generated codes in the following JSON list format:

```
{"task_id": "../..", "generation: "..."}
{"task_id": "../..", "generation: "..."}
...
```

and evaluate them using the following script under the root directory of the repository (<font color='red'>please execute with caution, the generated codes might have unexpected behaviours though with very low possibility. See the warnings in [execution.py](execution.py) and uncomment the execution lines at your own risk</font>):

```bash
bash scripts/evaluate_humaneval_x.sh <RESULT_FILE> <LANG> <N_WORKERS>
```
