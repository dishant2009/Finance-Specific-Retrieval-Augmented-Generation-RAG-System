"""
Setup script for Financial RAG System
Install with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="financial-rag-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A sophisticated RAG system for financial document question-answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/financial-rag-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial Institutions",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "finrag=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json"],
    },
    exclude_package_data={
        "": ["*.log", "*.pkl", "*.npy"],
    },
    keywords="rag, retrieval-augmented-generation, finbert, finance, nlp, question-answering",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/financial-rag-system/issues",
        "Source": "https://github.com/yourusername/financial-rag-system",
        "Documentation": "https://github.com/yourusername/financial-rag-system/wiki",
    },
)