# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

import versioneer

with open('README.md') as f:
    readme = f.read()

with open('HISTORY.md') as f:
    history = f.read()


requirements = [
    # to specify what a project minimally needs to run correctly
]


setup(
    name="magic_nlp",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Do some cool things with NLP",
    long_description=readme + '\n\n' + history,
    author="MrJStyle",
    author_email="438747096lmz@gmail.com",
    keywords="magic_nlp",
    url="https://www.mrjstyle.cn/",
    include_package_data=True,
    packages=find_packages(include=["magic_nlp", "magic_nlp.*"]),
    install_requires=requirements,
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [

        ]
    },
)
