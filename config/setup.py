"""
Setup script for CaMS (Calibrated Meta-Selection)
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CaMS: Calibrated Meta-Selection for Time Series Forecasting"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    return []

setup(
    name="cams",
    version="0.1.0",
    author="CaMS Contributors",
    author_email="",
    description="CaMS: Calibrated Meta-Selection framework for time series forecasting",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/cams",
    packages=find_packages(include=("cams", "cams.*")),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    keywords=[
        "machine learning",
        "deep learning",
        "time series",
        "forecasting",
        "meta-learning",
        "reinforcement learning",
        "uncertainty",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/cams/issues",
        "Source": "https://github.com/your-org/cams",
        "Documentation": "https://github.com/your-org/cams/blob/main/README.md",
        "Paper": "",
    },
)
