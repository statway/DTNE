# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    README = f.read()

setup(
    name='dtne',
    version='0.1.1',
    description='Implementation of Diffusion k-Nearest Neighbors',
    long_description=README,
    packages=find_packages(),
    author='jiangyong Wei',
    author_email='weijy026a@outlook.com',
    url='https://github.com/statway/dtne',
    install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib'], # 'pandas'
)
