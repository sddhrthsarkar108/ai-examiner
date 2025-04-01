import os
import sys
import logging
import platform
import base64
from pathlib import Path
import shutil

# Configure logging
def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def detect_poppler_path():
    """Detect poppler path based on operating system."""
    system = platform.system()
    if system == "Darwin":  # macOS
        for path in ["/opt/homebrew/bin/", "/usr/local/bin/"]:
            if os.path.exists(path):
                return path
    elif system == "Linux":
        for path in ["/usr/bin/", "/usr/local/bin/"]:
            if os.path.exists(path):
                return path
    elif system == "Windows":
        # Add Windows paths if needed
        return None
    
    logger.warning("Could not detect poppler path. Attempting to use system path.")
    return None

def encode_image_to_base64(image_path):
    """Convert an image file to base64 format."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def validate_pdf_path(pdf_path):
    """Validate the PDF path exists and is a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"The file {pdf_path} is not a PDF file")
    
    return pdf_path

def create_output_directory(pdf_path, app_config=None):
    """Create an output directory for this PDF run."""
    # Get base filename without extension
    base_name = os.path.basename(pdf_path).split('.')[0]
    
    # Get output directory from app config or use default
    output_base = "output"
    if app_config and "general" in app_config:
        output_base = app_config["general"].get("output_directory", "output")
    
    # Create base directory for this PDF
    base_output_dir = Path(output_base) / base_name
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Determine the next run number
    existing_runs = [d for d in os.listdir(base_output_dir) 
                   if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith("run_")]
    
    if existing_runs:
        # Extract run numbers and find the highest
        run_numbers = [int(run.split("_")[1]) for run in existing_runs]
        next_run = max(run_numbers) + 1
    else:
        next_run = 1
    
    # Create run directory
    run_dir = base_output_dir / f"run_{next_run}"
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

def create_ocr_directory(pdf_path, app_config=None):
    """Create a directory for OCR processing results."""
    # Get base filename without extension
    base_name = os.path.basename(pdf_path).split('.')[0]
    
    # Get output directory from app config or use default
    output_base = "output"
    if app_config and "general" in app_config:
        output_base = app_config["general"].get("output_directory", "output")
    
    # Create base directory for this PDF
    base_output_dir = Path(output_base) / base_name
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create OCR directory
    ocr_dir = base_output_dir / "ocr"
    os.makedirs(ocr_dir, exist_ok=True)
    
    return ocr_dir

def create_evaluation_directory(run_dir, eval_run):
    """Create directory for an evaluation run"""
    eval_dir = run_dir / f"evaluate_{eval_run}"
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir 