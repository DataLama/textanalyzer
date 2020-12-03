## 점차적으로 dependency좀 구현

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textanalyzer",
    version="0.0.5",
    author="Kim DongWook",
    author_email="kdw4521@gmail.com",
    description="Text Data Analysis module for analyzing text data in tabular data.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="TextMining NLP",
    license="MIT",
    url="https://github.com/DataLama/textanalyzer",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        # dataclasses for Python versions that don't have it
        "dataclasses;python_version<'3.7'",
        # utilities from PyPA to e.g. compare versions
        "packaging",
        # filesystem locks e.g. to prevent parallel downloads
        "filelock",
        # for downloading models over HTTPS
        "requests",
        # progress bars in model download and training scripts
        "tqdm >= 4.27",
        # tokenizers text preprocessing
        "emoji",
        "paddlepaddle==1.7.2",
        "lac",
        "soynlp",
        "lxml",
        "beautifulsoup4"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.0',
)