from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="prompt-engineering-pipeline",
    version="1.0.0",
    author="Advanced Prompting Research",
    author_email="research@promptengineering.ai",
    description="Multi-Path Reasoning Pipeline with Tree-of-Thought and Automated Prompt Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/prompt-engineering-pipeline",
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
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.8.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prompt-pipeline=src.pipeline:main",
            "prompt-eval=evaluation.evaluator:main",
        ],
    },
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
)