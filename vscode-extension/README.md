![codegeex_logo](../resources/logo/codegeex_logo.png)

🌐 <a href="https://github.com/THUDM/CodeGeeX/blob/main/vscode-extension/README_zh.md" target="_blank">中文</a>

![CodeGeeX vscode extension version](https://img.shields.io/visual-studio-marketplace/v/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX vscode extension last update](https://img.shields.io/visual-studio-marketplace/last-updated/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX download](https://img.shields.io/visual-studio-marketplace/d/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX vscode extension rating](https://img.shields.io/visual-studio-marketplace/stars/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX github stars](https://img.shields.io/github/stars/THUDM/CodeGeeX?style=social)

We introduce CodeGeeX, a large-scale multilingual code generation model with 13 billion parameters, pretrained on a large code corpus of more than 20 programming languages. With CodeGeeX, we can generate codes by only providing natural language descriptions, complete any code snippet, or translate codes to other programming languages, etc. CodeGeeX also provides customizable features (**Prompt Mode**) to help you configure your own programming assistant. Happy coding!

Please search "codegeex" in VS Code Marketplace to download it for free (minimum version requirement of VS Code is 1.68.0). For more information, please check out our [Homepage](https://models.aminer.cn/codegeex/) and [GitHub repo](https://github.com/THUDM/CodeGeeX).

Please kindly let us know if you encounter any problem or have any suggestion, via [codegeex@aminer.cn](mailto:codegeex@aminer.cn).

- [Basic Usage](#basic-usage)
- [Privacy](#privacy)
- [Guidance](#guidance)
  - [Stealth mode](#stealth-mode)
  - [Interactive mode](#interactive-mode)
  - [Translation mode](#translation-mode)
  - [Prompt mode (Experimental)](#prompt-mode-experimental)

## Basic Usage
Make sure vscode version is >= 1.68.0. Install the extension and enable it globally. There are four modes of usage:

-   **Stealth mode**: Keep CodeGeeX activated, it will start generating codes when you stop writing (the icon at the bottom of VSCode starts spinning). When the generated code is shown in gray, just press ``Tab`` to insert the generated codes. 
-   **Interactive mode**: Press ``Ctrl+Enter`` to activate the interactive mode, CodeGeeX will generate ``X`` candidates and show them in the right panel (``X`` can be modified in extension settings ``Candidate Num``). Then, select the best candidate by clicking on it.
-   **Translation mode**: Select code, and press ``Ctrl+Alt+T`` to activate the translation mode. Then, choose the language of the selected code. You will get the code translated into the same language as your current editor. Click on the ``use code`` button to insert the result. You can also configure in the settings whether to comment out the original code or to replace it.
-   **Prompt mode (experimental)**: Select codes to be used as input, then press ``Alt/Option+t`` to trigger the prompt mode. It will show a list of pre-defined prompt templates and choose one to generate codes with your input. This mode is fully customizable, you can add your own templates in the extension settings ``Prompt Templates``. 

## Privacy
We highly respect the privacy of your code. The code is only used as the input of CodeGeeX to assist your programming. At the first time of usage, we will ask if you agree to share the generated code only for research purpose (**disabled** by default).

## Guidance
Please see the details and examples for how to use the three modes in CodeGeeX:
### Stealth mode
In this mode, CodeGeeX will start generating codes when you stop writing (the icon at the bottom of VSCode starts spinning). When the generated code is shown in gray, just press ``Tab`` to insert the generated codes. You can also press ``Alt/Option+[`` or ``]`` to change between candidates. If you are not satisfied with the current suggestions, you can also press ``Alt/Option+N`` to get new suggestions. Change the number of candidates in the extension settings ``Candidate Num`` (more candidates will slow down the generation speed). **Note**: The generation always starts at the current position of your cursor, thus if you modify the code before the generation is finished, it will probably cause bugs. We keep working on making the generation faster.

![image](https://lfs.aminer.cn/misc/wangshan/pretrain/codegeex/bubble_sort_go.gif)

### Interactive mode
In this mode, press ``Ctrl+Enter`` to generate codes and visualize the candidates in another panel. Then, click on the best candidate to insert the generated codes to the current position of cursor. 

![image](https://lfs.aminer.cn/misc/wangshan/pretrain/codegeex/interactive_mode2.gif)

### Translation mode
In this mode, paste or tape some code from another language to the current file, select the code, and press ``Ctrl+Alt+T``. Then, choose the language of the selected code. Wait for a few seconds, you will get the code translated into the same language as your current editor. Click on the ``use code`` button to insert the result into the current position of your cursor. You can also configure in the settings whether to comment out the original code or to replace it.

![image](https://lfs.aminer.cn/misc/wangshan/pretrain/codegeex/translation_cpp_to_python.gif)

### Prompt mode (Experimental)
In this mode, you can add extra prompts to the input and implement some cool features, like code explanation, summarization, generation with specific coding style, and more. The principle behind is the few-shot ability of CodeGeeX. When you provide a few examples as extra prompts in the input, CodeGeeX will imitate what are done by these examples and generate codes accordingly. For example, you can give an example that explains each line of code. Select the code you want to explain, then press ``Alt/Option+t`` to trigger the prompt mode. It will show a list of pre-defined prompt templates and choose the ``explanation`` to generate codes with your input. Magically, the codes will be explained line by line.

![image](https://lfs.aminer.cn/misc/wangshan/pretrain/codegeex/explanation_python.gif)

The template of the above example looks like the following, which contains ``[Example code]``, ``<INPUT>``, ``[Example code with explanation]`` and ``[Explanation head]``. ``<INPUT>`` is where the selected code will be inserted. ``<INPUT0:1>`` means the first line of your input (which is used here to ensure the same function will be explained). When you use the prompt mode, CodeGeeX will combine your input with the template and use them all as the input to generate codes. 

```python
# language: Python

def sum_squares(lst):
    sum = 0
    for i in range(len(lst)):
        if i % 3 == 0:
            lst[i] = lst[i]**2
        elif i % 4 == 0:
            lst[i] = lst[i]**3
        sum += lst[i]
    return sum

<INPUT>

# Explain the code line by line
def sum_squares(lst):
    # initialize sum
    sum = 0
    # loop through the list
    for i in range(len(lst)):
        # if the index is a multiple of 3
        if i % 3 == 0:
            # square the entry
            lst[i] = lst[i]**2
        # if the index is a multiple of 4
        elif i % 4 == 0:
            # cube the entry
            lst[i] = lst[i]**3
        # add the entry to the sum
        sum += lst[i]
    # return the sum
    return sum

# Explain the code line by line
<INPUT:0,1>
```

And here is another example for python docstring generation    
```python
def add_binary(a, b):
    '''
    Returns the sum of two decimal numbers in binary digits.

    Parameters:
            a (int): A decimal integer
            b (int): Another decimal integer

    Returns:
            binary_sum (str): Binary string of the sum of a and b
    '''
    binary_sum = bin(a+b)[2:]
    return binary_sum

<INPUT>
```

The templates are fully customizable, you can add your own templates in the extension settings ``Prompt Templates``. ``key`` is the name that you want to show in the list of templates, ``value`` is the path to the template file (``.txt``, ``.py``, ``.h``, etc). Try this feature and write your own templates, you can make the generated codes follow your coding style, generate with a specific function name, or add a specific comment, etc.
