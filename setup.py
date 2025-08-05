"""
Setup script for Dynamic Information Lattices
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Dynamic Information Lattices: A New Paradigm for Efficient Generative Modeling"

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
    name="dynamic-info-lattices",
    version="1.0.0",
    author="Dynamic Information Lattices Team",
    author_email="contact@dynamic-info-lattices.org",
    description="A novel paradigm for efficient generative modeling through information-theoretic computational geometry",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/xiufengliu/dynamic_info_lattices",
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
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
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
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "fast": [
            "pyfftw>=0.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "dil-train=examples.train_dil:main",
            "dil-evaluate=examples.evaluate_dil:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dynamic_info_lattices": [
            "data/*.csv",
            "configs/*.yaml",
        ],
    },
    keywords=[
        "machine learning",
        "deep learning",
        "time series",
        "forecasting",
        "diffusion models",
        "information theory",
        "generative modeling",
        "pytorch"
    ],
    project_urls={
        "Bug Reports": "https://github.com/xiufengliu/dynamic_info_lattices/issues",
        "Source": "https://github.com/xiufengliu/dynamic_info_lattices",
        "Documentation": "https://github.com/xiufengliu/dynamic_info_lattices/blob/main/README.md",
        "Paper": "https://github.com/xiufengliu/dynamic_info_lattices",
    },
)
