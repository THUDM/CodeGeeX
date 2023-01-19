# encoding:utf-8

import json

import requests

'''
Code Generation
'''
API_KEY = ""  # Get from Tianqi console. 从控制台获取
API_SECRET = ""  # Get from Tianqi console. 从控制台获取
PROMPT = "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    " \
         "\"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given " \
         "threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements(" \
         "[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"
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
    "prompt": PROMPT,
    "n": NUMBER,
    "lang": LANG
}


def main():
    response = requests.post(request_url, headers=headers, data=json.dumps(data))
    if response:
        print(response.json())


if __name__ == '__main__':
    main()
