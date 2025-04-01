import json
import logging
import os
import re
import shutil
import time
from datetime import datetime

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from pdf2image import convert_from_path

from config import DefaultConfig
from llm_factory import LLMFactory
from prompts import get_ocr_system_prompt, get_ocr_user_prompt
from utils import detect_poppler_path, encode_image_to_base64

logger = logging.getLogger(__name__)


def create_ocr_chain(
    model_name="gpt-4o",
    temperature=0.0,
    max_tokens=4000,
    api_key=None,
    provider="openai",
):
    """Create a LangChain for OCR processing.

    Args:
        model_name: The name of the model to use
        temperature: Temperature setting for model output randomness
        max_tokens: Maximum tokens for model response
        api_key: API key for the model provider
        provider: The model provider to use (default: openai)

    Returns:
        A LangChain chain for OCR processing
    """
    # Use the LLM factory to create the OCR chain
    return LLMFactory.create_chain_for_ocr(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def extract_text_from_image(image_path, output_dir, app_config=None, api_key=None):
    """Extract handwritten text from an image using LangChain with vision models.

    Args:
        image_path: Path to the image file
        output_dir: Directory to save output files
        app_config: Optional application configuration dictionary
        api_key: API key for the model provider
    """
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return "Error: Could not encode image."

    # Apply custom config if provided, otherwise use default
    config = DefaultConfig.OCR.copy()
    if app_config and "ocr" in app_config:
        config = app_config["ocr"]

    # Get domain_context from general section
    domain_context = "handwritten answer sheet for a machine learning exam"
    if app_config and "general" in app_config:
        domain_context = app_config["general"].get(
            "domain_context", "machine learning exam"
        )

    # Save the API call request for audit purposes
    with open(output_dir / "api_request_log.txt", "a") as f:
        f.write(
            f"\n--- Image OCR Request: {image_path} at {datetime.now().isoformat()} ---\n"
        )

    # Get prompts from centralized prompts module
    include_examples = config.get("include_examples", False)
    system_prompt = get_ocr_system_prompt(domain_context)
    user_prompt = get_ocr_user_prompt(domain_context, include_examples)

    # Create a message with image for GPT-4o
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        ),
    ]

    try:
        # Add rate limiting
        time.sleep(1)  # 1 second delay between calls

        # Get configuration parameters
        temperature = config.get("temperature", 0.0)
        max_tokens = config.get("max_tokens", 4000)
        model_name = config.get("model", "gpt-4o")
        provider = config.get("provider", "openai")

        # Create the LangChain chain with the API key and provider
        chain = create_ocr_chain(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            provider=provider,
        )

        # Run the chain to get extracted text
        extracted_text = chain({"messages": messages})

        # Save extracted text for audit
        with open(output_dir / "extracted_text_log.txt", "a") as f:
            f.write(
                f"\n--- Extracted from {image_path} at {datetime.now().isoformat()} ---\n"
            )
            f.write(extracted_text)
            f.write("\n---\n")

        return extracted_text
    except Exception as e:
        logger.error(f"Error during OCR extraction: {e}")
        return f"Error: {str(e)}"


def extract_answers_from_pdf(pdf_path, output_dir, app_config=None, api_key=None):
    """Convert a multi-page scanned PDF to images and extract handwritten text.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        app_config: Optional application configuration dictionary
        api_key: API key for the model provider
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    temp_dir = output_dir / "temp_images"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        poppler_path = detect_poppler_path()
        conversion_args = {"poppler_path": poppler_path} if poppler_path else {}

        logger.info(f"Converting PDF using poppler path: {poppler_path}")
        images = convert_from_path(pdf_path, **conversion_args)

        extracted_answers = {}
        for i, image in enumerate(images):
            image_path = temp_dir / f"page_{i + 1}.jpg"
            image.save(image_path, "JPEG")

            logger.info(f"Extracting text from page {i + 1}...")
            extracted_text = extract_text_from_image(
                image_path, output_dir, app_config, api_key
            )
            extracted_answers[f"Page {i + 1}"] = extracted_text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temporary image files")

    # Save extracted answers to file
    with open(output_dir / "extracted_answers.json", "w") as f:
        json.dump(extracted_answers, f, indent=2)

    # Process the extracted answers to clean up formatting
    processed_answers = process_extracted_answers(extracted_answers)
    with open(output_dir / "processed_answers.json", "w") as f:
        json.dump(processed_answers, f, indent=2)

    return processed_answers


def process_extracted_answers(extracted_answers):
    """Process and clean up extracted answers to make them more consistent."""
    processed = {}

    for page, content in extracted_answers.items():
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

        processed[page] = content.strip()

    return processed
