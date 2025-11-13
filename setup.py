"""Setup configuration for ragcun package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ragcun",
    version="0.1.0",
    author="RAGCUN Team",
    description="A simple and effective framework for building RAG applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ctn/ragcun",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "nltk>=3.8.0",
        "pypdf>=3.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "openai>=1.0.0",
        ],
    },
)
