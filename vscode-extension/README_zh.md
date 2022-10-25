![codegeex_logo](../resources/logo/codegeex_logo.png)

🌐 <a href="https://github.com/THUDM/CodeGeeX/blob/main/vscode-extension/README.md" target="_blank">English</a>

![CodeGeeX vscode extension version](https://img.shields.io/visual-studio-marketplace/v/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX vscode extension last update](https://img.shields.io/visual-studio-marketplace/last-updated/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX download](https://img.shields.io/visual-studio-marketplace/d/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX API calls total](https://img.shields.io/badge/dynamic/json?colorA=0B9FE0&colorB=brightgreen&label=API%20calls&query=%24.result.count&url=https%3A%2F%2Ftianqi.aminer.cn%2Fapi%2Fv1%2Fapi%2Fcodegeex%2Fdashboard%3Ftime_type%3Dall%26apiName%3Dall%26num%3D1)
![CodeGeeX vscode extension rating](https://img.shields.io/visual-studio-marketplace/stars/aminer.codegeex?colorA=0B9FE0&colorB=brightgreen)
![CodeGeeX github stars](https://img.shields.io/github/stars/THUDM/CodeGeeX?style=social)

CodeGeeX是一个具有130亿参数的多编程语言代码生成预训练模型，使用超过二十种编程语言训练得到。基于CodeGeeX开发的插件可以实现通过描述生成代码、补全代码、代码翻译等一系列功能。CodeGeeX同样提供可以定制的**提示模式（Prompt Mode）**，构建专属的编程助手。Happy Coding！

VS Code插件市场搜索"codegeex"即可免费使用(需要VS Code版本不低于1.68.0)，更多关于CodeGeeX信息请见我们的[主页](https://models.aminer.cn/codegeex/) and [GitHub仓库](https://github.com/THUDM/CodeGeeX)。

如使用过程中遇到问题或有任何改进意见，欢迎发送邮件到[codegeex@aminer.cn](mailto:codegeex@aminer.cn)反馈！

## 基本用法
安装插件并全局激活CodeGeeX，有以下三种使用模式：

-   **隐匿模式**: 保持CodeGeeX处于激活状态，当您停止输入时，会从当前光标处开始生成（右下角CodeGeeX图标转圈表示正在生成）。 生成完毕之后会以灰色显示，按``Tab``即可插入生成结果。 
-   **交互模式**: 按``Ctrl+Enter``激活交互模式，CodeGeeX将生成``X``个候选，并显示在右侧窗口中（``X`` 数量可以在设置的``Candidate Num``中修改）。 点击候选代码上方的``use code``即可插入。
-   **提示模式**: 选择需要作为输入的代码，按``Alt/Option+t``触发提示模式，会显示预定义模板列表，选择其中一个模板，即可将代码插入到模板中进行生成。 这个模式高度自定义，可以在设置中 ``Prompt Templates``修改或添加模板内容，为模型加入额外的提示。 

## 隐私声明

我们高度尊重用户代码的隐私，代码仅用来辅助编程。在您第一次使用时，我们会询问您是否同意将生成的代码用于研究用途，帮助CodeGeeX变得更好（该选项默认**关闭**）。

## 使用指南

以下是CodeGeeX几种模式的详细用法：

### 隐匿模式

在该模式中，CodeGeeX将在您停止输入时，从光标处开始生成（右下角CodeGeeX图标转圈表示正在生成）。生成完毕之后会以灰色显示，按``Tab``即可插入生成结果。 在生成多个候选的情况下，可以使用``Alt/Option+[`` 或 ``]``在几个候选间进行切换。可以在设置中改变``Candidate Num``（增加个数会导致生成速度相对变慢）。**注意**：生成总是从当前光标位置开始，如果您在生成结束前移动光标位置，可能会导致一些bugs。我们正在努力使生成速度变得更快以提升用户体验。

![image](stealth_mode.gif)

### 交互模式

在该模式中，按``Ctrl+Enter``激活交互模式，CodeGeeX将生成``X``个候选，并显示在右侧窗口中（``X`` 数量可以在设置的``Candidate Num``中修改）。 点击候选代码上方的``use code``即可插入结果到为当前光标位置。 

![image](interactive_mode.gif)

### 提示模式

在该模式中，您可以在输入中添加额外的提示来实现一些有趣的功能，包括并不限于代码解释、概括、以特定风格生成等。该模式的原理是利用了CodeGeeX强大的少样本生成能力。当您在输入中提供一些例子时，CodeGeeX会模仿这些例子并实现相应的功能。比如，您可以自定义模板中提供一段逐行解释代码的例子。选择您想要解释的代码，按``Alt/Option+t``触发提示模式，选择您写好的模板（如``explanation``），CodeGeeX就会解释您输入的代码。以下我们会详细介绍如何制作模板。

![image](prompt_mode.gif)

上述例子中的模板如下图所示，由``[示例代码]``, ``<INPUT>``, ``[带解释的示例代码]`` and ``[输出函数头]`` 。``<INPUT>``表示您选中的代码将会插入的位置。 ``<INPUT:0,1>`` 这一句用来保证模型解释的是同一个函数。当使用提示模式时，CodeGeeX会将您选择的代码（插入到<INPUT>部分）和模板代码相结合，一起作为模型的输入。 

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

以下是另一个Python文档字符串生成的例子，CodeGeeX在您写新函数时会模仿该注释的格式：
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

模板文件是高度自定义化的，您可以将自定义模板添加到插件设置中的``Prompt Templates``中。 ``key``表示模板的名字， ``value``是模板文件的路径（可以是您电脑上的任一路径，``.txt``, ``.py``, ``.h``, 等格式文件均可）。通过该功能，您可以让CodeGeeX生成具有特定风格或功能的代码，快尝试定义自己的专属模板吧！
