# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


# Read the contents of your README file (optional, useful for PyPI description)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dtne',
    version='0.5',
    description='Implementation of diffusive topology neighbor embedding',
    # long_description=README,
    packages=find_packages(),
    author='jiangyong Wei',
    author_email='weijy026a@outlook.com',
    url='https://github.com/statway/dtne',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib'],
    classifiers=[  # See https://pypi.org/classifiers/ for all classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Specify the Python version requirement
)