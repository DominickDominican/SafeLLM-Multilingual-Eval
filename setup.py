from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="safellm-multilingual-eval",
    version="0.1.0",
    author="Dominick Dominican",
    author_email="dominickdominican47@gmail.com",
    description="A multilingual evaluation framework for testing LLM safety, robustness, and alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DominickDominican/SafeLLM-Multilingual-Eval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
        "Topic :: Security",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
    },
    entry_points={
        "console_scripts": [
            "safellm-eval=safellm_eval.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "safellm_eval": ["datasets/*.jsonl", "config/*.yaml"],
    },
)