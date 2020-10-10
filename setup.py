#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name='mlm',
    version='0.1',
    description="Masked Language Model Scoring",
    author='Julian Salazar',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    entry_points = {
        'console_scripts': ['mlm=mlm.cmds:main'],
    },

    install_requires=[
        'gluonnlp~=0.8.3',
        'regex',
        'sacrebleu',
        'mosestokenizer',
        'transformers~=3.3.1'
    ],

    extras_require={
        'dev': [
            'pylint',
            'pytest',
            'pytest-cov',
            'mypy'
        ]
    },

    # Needed for static type checking
    # https://mypy.readthedocs.io/en/latest/installed_packages.html
    zip_safe=False
)
