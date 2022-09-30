![codegeex_logo](../resources/logo/codegeex_logo.png)

# 创建CodeGeeX API

使用[天启 · API开放平台](https://tianqi.aminer.cn/open/)申请CodeGeeX API：

<img src="../resources/api/api_step_1.png">

点击首页中的天启平台体验入口：
<img src="../resources/api/api_step_2.png">
点击API应用：
<img src="../resources/api/api_step_3.png">
输入任意名称，创建API应用。创建后会得到API Key/Secret，用于调用API：
<img src="../resources/api/api_step_4.png">

在API信息中，可以查看代码生成/代码翻译的请求地址和使用文档：
<img src="../resources/api/api_step_5.png">

根据文档中的描述使用API，参考文件``api/generation_example.py``：

```python
# encoding:utf-8

import requests
import json

'''
Code Generation
'''
API_KEY = ""  # Get from Tianqi console. 从控制台获取
API_SECRET = ""  # Get from Tianqi console. 从控制台获取
PROMPT = "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
NUMBER = 3
LANG = "Python"
request_url = "https://tianqi.aminer.cn/api/v2/"
api = 'multilingual_code_generate'

# Request is in json format. 指定请求参数格式为json
headers = {'Content-Type': 'application/json'}
request_url = request_url + api
data = {
    "apikey": API_KEY,
    "apisecret": API_SECRET,
    "prompt":PROMPT,
    "n":NUMBER,
    "lang":LANG
}

def main():
    response = requests.post(request_url, headers=headers, data=json.dumps(data))
    if response:
        print(response.json())

if __name__ == '__main__':
    main()
```

