#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name='intpiece',
    install_requires=['numpy', 'tqdm', "ahocorasick"],
    packages=find_packages(),
    ext_modules=cythonize('*.pyx'),
    package_data={'intpiece': ['*.pyx']},
    include_package_data=True
)
