"""Setup script for the automated answer script package."""

from setuptools import find_packages, setup

# Define version at the module level for use by pyproject.toml
__version__ = "0.2.0"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="automated-answer-script",
    version=__version__,
    author="Siddhartha Sarkar",
    author_email="sidsar@duck.com",
    description="An AI-powered tool for extracting, interpreting, and evaluating handwritten exam answers using OCR and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sddhrthsarkar108/ai-examiner",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Core dependencies
        "openai>=1.6.1,<2.0.0",
        "python-dotenv==1.0.0",
        "requests==2.32.3",
        "pydantic>=2.5.2,<3.0.0",
        
        # Data processing
        "pandas>=2.2.0,<2.3.0",
        "numpy>=1.26.4,<2.0.0",
        
        # PDF processing
        "pdf2image==1.16.3",
        "Pillow==9.5.0",
        
        # LangChain dependencies
        "langchain>=0.3.22",
        "langchain-openai>=0.3.11",
        "langchain-deepseek>=0.1.3",
        "langchain-google-genai>=2.1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Education :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="education, exam, grading, OCR, AI, LLM, handwriting, evaluation",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "answer-script=main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/sddhrthsarkar108/ai-examiner/issues",
        "Documentation": "https://github.com/sddhrthsarkar108/ai-examiner#readme",
    },
)

# Note: This package requires the system-level dependency 'poppler-utils'
# On Ubuntu/Debian: sudo apt-get install poppler-utils
# On macOS: brew install poppler
# On Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
