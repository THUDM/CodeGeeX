# HumanEval-X: 多语言代码生成基准

🌐 <a href="README.md" target="_blank">English</a>

为了更好地评测代码生成模型的多语言生成能力，我们构建了一个新基准HumanEval-X。此前，多语言代码生成能力是基于语义相似度（比如[CodeBLEU](https://arxiv.org/abs/2009.10297)）衡量的，具有一定误导性；HumanEval-X则可用于衡量生成代码的功能正确性。HumanEval-X包含820个高质量手写样本，覆盖Python、C++、Java、JavaScript、Go，可用于多种任务。

<img src="../../resources/en/hx_tasks.png">

<p align="center"><i><b>HumanEval-X</b>支持的任务示例。<font style='background-color:#F8CECC'>声明</font>、<font style='background-color:#D5E8D4'>描述</font>、<font style='background-color:#DAE8FC'>解答</font>分别用红、绿、蓝色标注。<i>代码生成</i>将声明与描述作为输入，输出解答。<i>代码翻译</i>将两种语言的声明与源语言的解答作为输入，输出目标语言的解答。</i></p>

HumanEval-X中每个语言的样本，包含了声明、描述和解答，它们之间的组合可以支持不同的下游任务，包括生成、翻译、概括等。我们目前关注两个任务：**代码生成**与**代码翻译**。对于代码生成任务，模型将函数声明与文档字符串作为输入，输出函数实现；对于代码翻译任务，模型将两种语言的函数声明与源语言的实现作为输入，输出目标语言上的实现。我们在代码翻译任务中不将文档字符串输入模型，以避免模型直接通过描述生成答案。在两种任务下，我们都采用[Codex](https://arxiv.org/abs/2107.03374)所使用的无偏pass@k指标：$\text{pass}@k:= \mathbb{E}[1-\frac{\tbinom{n-c}{k}}{\tbinom{n}{k}}]$, $n=200$, $k\in(1,10,100)$。

## 如何使用HumanEval-X

样本使用JSON列表格式存储在``codegeex/benchmark/humaneval-x/[LANG]/data/humaneval_[LANG].jsonl.gz``，每条样本包含6个部分：

*   ``task_id``: 题目的目标语言与ID。语言为["Python", "Java", "JavaScript", "CPP", "Go"]中之一。
*   ``prompt``: 函数声明与描述，用于代码生成。
*   ``declaration``: 仅有函数声明，用于代码翻译。
*   ``canonical_solution``: 手写的示例解答。
*   ``test``: 隐藏测例，用于评测。
*   ``example_test``: 提示中出现的公开测例，用于评测。

### 评测环境

评测生成的代码需要使用多种语言编译、运行。我们使用的各编程语言依赖及所用包的版本如下：

| 依赖    | 版本     |
| ------- | -------- |
| Python  | 3.8.12   |
| JDK     | 18.0.2.1 |
| Node.js | 16.14.0  |
| js-md5  | 0.7.3    |
| C++     | 11       |
| g++     | 7.5.0    |
| Boost   | 1.71.0   |
| OpenSSL | 3.0.0    |
| go      | 1.18.4   |

为了省去使用者配置这些语言环境的麻烦，我们构建了一个Docker镜像，并在其中配置了所需要的环境。

可以直接从Docker Hub拉取镜像：

```bash
docker pull rishubi/codegeex:latest
```

如果您熟悉Dockerfile，也可以从`codegeex/docker/Dockerfile`构建镜像，或者修改之以定制自己的配置：

```bash
cd codegeex/docker
docker build [OPTIONS] .
```

获取镜像后，使用如下命令创建容器：

```bash
docker run -it --gpus all --mount type=bind,source=<LOCAL PATH>,target=<PATH IN CONTAINER> [OPTIONS] <IMAGE NAME:TAG>
```

### 评测

我们推荐使用给定的[评测环境](#评测环境)进行评测。在评测前，将生成的代码以如下JSON列表形式存储：

```
{"task_id": "../..", "generation: "..."}
{"task_id": "../..", "generation: "..."}
...
```

并在本仓库的根目录下使用如下指令（<font color='red'>请谨慎执行，生成的代码可能有极低概率产生意外行为。在[execution.py](execution.py)中查看警告并取消执行代码的注释，风险自负</font>）：

```bash
bash scripts/evaluate_humaneval_x.sh <RESULT_FILE> <LANG> <N_WORKERS>
```
