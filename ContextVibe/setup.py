#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContextVibe 包安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "ContextVibe: 多模态会议情感分析系统"

# 读取requirements文件
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="contextvibe",
    version="1.0.0",
    author="ContextVibe Team",
    author_email="contact@contextvibe.com",
    description="多模态会议情感分析系统",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/contextvibe/contextvibe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "contextvibe=contextvibe.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "contextvibe": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="multimodal, emotion, analysis, meeting, audio, video, text, vaec",
    project_urls={
        "Bug Reports": "https://github.com/contextvibe/contextvibe/issues",
        "Source": "https://github.com/contextvibe/contextvibe",
        "Documentation": "https://contextvibe.readthedocs.io/",
    },
)
