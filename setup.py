"""Setup script for the automated answer script package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="automated-answer-script",
    version="0.1.0",
    author="Siddhartha Sarkar",
    author_email="sidsar@duck.com",
    description="A tool for extracting and evaluating handwritten answers from exam sheets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sddhrthsarkar108/ai-examiner",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai>=1.0.0",
        "pdf2image>=1.16.3",
        "pillow>=9.4.0",
        "python-dotenv>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "answer-script=src.main:main",
        ],
    },
) 