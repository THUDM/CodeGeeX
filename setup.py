from setuptools import setup, find_packages

setup(
    name="codegeex",
    py_modules=["codegeex"],
    version="1.0",
    description="CodeGeeX: A Open Multilingual Code Generation Model.",
    author="Qinkai Zheng",
    packages=find_packages(),
    install_requires=[
        "fire>=0.4.0",
        "ipython>=8.4.0",
        "numpy>=1.22.0",
        "pandas>=1.3.5",
        "pyzmq>=23.2.1",
        "regex>=2022.3.15",
        "setuptools>=58.0.4",
        "transformers>=4.22.0,<=4.24.0",
        "tokenizers>=0.11.0,<=0.11.4",
        "pytorch>=1.10.0",
        "tqdm>=4.63.0",
    ],
    entry_points={}
)
