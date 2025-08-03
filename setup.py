"""
PPG血压预测系统安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "PPG血压预测系统"

# 读取requirements文件
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="ppg-bp-prediction",
    version="1.0.0",
    author="PPG-BP Team",
    author_email="team@ppg-bp.com",
    description="基于PPG信号的血压预测系统",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ppg-bp/ppg-bp-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
            "torchvision>=0.10.0+cu111",
        ]
    },
    entry_points={
        "console_scripts": [
            "ppg-bp-train=src.training.train:main",
            "ppg-bp-predict=src.inference.predict:main",
            "ppg-bp-evaluate=src.evaluation.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.yaml", "config/*.json"],
        "data": ["*.csv", "*.json"],
        "models": ["*.pth", "*.pkl"],
    },
    zip_safe=False,
    keywords=[
        "ppg", "blood pressure", "machine learning", "deep learning", 
        "healthcare", "biomedical signal processing", "cardiovascular"
    ],
    project_urls={
        "Bug Reports": "https://github.com/ppg-bp/ppg-bp-prediction/issues",
        "Source": "https://github.com/ppg-bp/ppg-bp-prediction",
        "Documentation": "https://ppg-bp-prediction.readthedocs.io/",
    },
)
