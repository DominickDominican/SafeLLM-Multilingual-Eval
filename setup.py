#!/usr/bin/env python3
"""
Setup script for SafeLLM Multilingual Evaluation Framework
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="safellm-multilingual-eval",
    version="1.0.0",
    author="Dominick Dominican",
    author_email="dominickdominican47@gmail.com",
    description="A multilingual evaluation framework for testing LLM safety, robustness, and alignment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "nbformat>=5.9.0",
            "ipywidgets>=8.0.0",
        ],
        "advanced": [
            "scipy>=1.11.0",
            "scikit-learn>=1.3.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "safellm-eval=safellm_eval.evaluator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "safellm_eval": ["datasets/*.jsonl", "config/*.yaml"],
    },
    keywords="llm, multilingual, safety, evaluation, ai-alignment, adversarial-testing",
)