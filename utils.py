import base64
import json
import logging
import os
import platform
import re
from pathlib import Path


# Configure logging
def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
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

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"The file {pdf_path} is not a PDF file")

    return pdf_path


def validate_text_path(text_path):
    """Validate the text file path exists and is a text file."""
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"File not found: {text_path}")

    if not text_path.lower().endswith(".txt"):
        raise ValueError(f"The file {text_path} is not a text file")

    return text_path


def load_ocr_from_text_file(text_path, output_dir):
    """Load OCR results from a text file.

    Args:
        text_path: Path to the text file
        output_dir: Directory to save output files

    Returns:
        Dictionary of OCR output by page
    """
    logger.info(f"Loading OCR results from text file: {text_path}")

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Process the content to create a page-based dictionary
        # Look for standard page markers like "=== Page X ==="
        page_pattern = re.compile(r"===\s*Page\s+(\d+)\s*===", re.MULTILINE)
        page_matches = list(page_pattern.finditer(content))

        ocr_results = {}

        if page_matches:
            # If page markers found, split content by those markers
            for i, match in enumerate(page_matches):
                page_num = match.group(1)
                start_pos = match.end()

                # Find end position (start of next page or end of file)
                if i < len(page_matches) - 1:
                    end_pos = page_matches[i + 1].start()
                else:
                    end_pos = len(content)

                page_content = content[start_pos:end_pos].strip()
                ocr_results[f"Page {page_num}"] = page_content
        else:
            # Try alternative pattern: look for "Page X" markers
            alt_page_pattern = re.compile(r"^Page\s+(\d+)[:\s]*$", re.MULTILINE)
            alt_page_matches = list(alt_page_pattern.finditer(content))

            if alt_page_matches:
                # If alternative page markers found, split content by those markers
                for i, match in enumerate(alt_page_matches):
                    page_num = match.group(1)
                    start_pos = match.end()

                    # Find end position (start of next page or end of file)
                    if i < len(alt_page_matches) - 1:
                        end_pos = alt_page_matches[i + 1].start()
                    else:
                        end_pos = len(content)

                    page_content = content[start_pos:end_pos].strip()
                    ocr_results[f"Page {page_num}"] = page_content
            else:
                # If no page markers found, treat the entire file as one page
                ocr_results["Page 1"] = content.strip()

        # Save the loaded content to the OCR output directory
        with open(output_dir / "extracted_answers.json", "w") as f:
            json.dump(ocr_results, f, indent=2)

        # Process and save formatted answers
        processed_answers = {}
        for page, content in ocr_results.items():
            # Clean up any double spaces or excessive newlines
            content = re.sub(r"\n{3,}", "\n\n", content)
            content = re.sub(r" {2,}", " ", content)

            # Ensure standardization of continuation markers
            content = re.sub(
                r"\[CONTINUES_FROM_PREVIOUS\]", "[CONTINUES_FROM_PREVIOUS]\n", content
            )
            content = re.sub(r"\[CONTINUES_TO_NEXT\]", "\n[CONTINUES_TO_NEXT]", content)

            # Ensure proper spacing around question markers
            content = re.sub(r"\[Q([0-9]+)\]", r"\n[Q\1]\n", content)

            # Remove any duplicate newlines created
            content = re.sub(r"\n{3,}", "\n\n", content)

            processed_answers[page] = content.strip()

        # Save processed answers
        with open(output_dir / "processed_answers.json", "w") as f:
            json.dump(processed_answers, f, indent=2)

        return processed_answers
    except Exception as e:
        logger.error(f"Error loading OCR from text file: {e}")
        raise


def create_output_directory(pdf_path, app_config=None):
    """Create an output directory for this PDF run."""
    # Get base filename without extension
    base_name = os.path.basename(pdf_path).split(".")[0]

    # Get output directory from app config or use default
    output_base = "output"
    if app_config and "general" in app_config:
        output_base = app_config["general"].get("output_directory", "output")

    # Create base directory for this PDF
    base_output_dir = Path(output_base) / base_name
    os.makedirs(base_output_dir, exist_ok=True)

    # Determine the next run number
    existing_runs = [
        d
        for d in os.listdir(base_output_dir)
        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith("run_")
    ]

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
    base_name = os.path.basename(pdf_path).split(".")[0]

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
