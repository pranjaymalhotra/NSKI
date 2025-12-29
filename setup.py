"""
NSKI: Neural Surgical Key-Value Intervention

Setup script for pip installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ]

setup(
    name="nski",
    version="1.0.0",
    author="Pranjay Malhotra",
    author_email="pranjay@example.com",
    description="Neural Surgical Key-Value Intervention for LLM Safety",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pranjaymalhotra/NSKI",
    project_urls={
        "Bug Tracker": "https://github.com/pranjaymalhotra/NSKI/issues",
        "Documentation": "https://github.com/pranjaymalhotra/NSKI#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "full": [
            "bitsandbytes>=0.42.0",
            "accelerate>=0.25.0",
            "datasets>=2.15.0",
            "scikit-learn>=1.3.0",
            "scipy>=1.11.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nski=nski.cli:main",
            "nski-experiment=nski.experiments.run_all:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nski": ["configs/*.yaml"],
    },
)
