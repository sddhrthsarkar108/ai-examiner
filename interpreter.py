import os
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Dict, List, Optional, Any

from config import DefaultConfig
from llm_factory import LLMFactory
from prompts import get_interpreter_system_prompt, get_interpreter_user_prompt

logger = logging.getLogger(__name__)

class QuestionContent(BaseModel):
    """Structure for content within a question."""
    question_number: str = Field(description="The question number")
    raw_text: str = Field(description="The raw text of the answer")
    equations: List[str] = Field(default_factory=list, description="List of equations in the answer")
    figures: List[str] = Field(default_factory=list, description="List of figures in the answer")
    calculations: List[str] = Field(default_factory=list, description="List of calculations in the answer")
    continues_from_previous: bool = Field(default=False, description="Whether this continues from a previous page")
    continues_to_next: bool = Field(default=False, description="Whether this continues to the next page")

class InterpretedAnswer(BaseModel):
    """Structure for the full interpreted answer set."""
    questions: Dict[str, QuestionContent] = Field(description="Dictionary of questions keyed by question number")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the answer set")
    summary: str = Field(default="", description="Summary of the answer content")

def create_interpreter_chain(model_name="gemini-2.5-pro", temperature=0.1, max_tokens=4000, api_key=None, provider="google"):
    """Create a chain for interpreting OCR output.
    
    Args:
        model_name: The name of the model to use
        temperature: Temperature setting for model output randomness
        max_tokens: Maximum tokens for model response
        api_key: API key for the model provider
        provider: The model provider to use (default: google)
        
    Returns:
        A chain for interpreting OCR output
    """
    # Use the LLM factory to create the model
    model = LLMFactory.create_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
    
    # Create a chain that interprets the OCR output
    def run_chain(ocr_data):
        # Get the system prompt from centralized prompts
        system_prompt = get_interpreter_system_prompt()
        
        # Create a message with OCR data
        pages_content = []
        for page_number, content in ocr_data.items():
            pages_content.append(f"--- {page_number} ---\n{content}")
        
        all_content = "\n\n".join(pages_content)
        
        # Get the user prompt from centralized prompts
        user_prompt = get_interpreter_user_prompt(all_content)
        
        try:
            # Add rate limiting
            time.sleep(1)  # 1 second delay between calls
            
            # Create messages for the model
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Invoke the model
            result = model.invoke(messages)
            result_text = result.content
            
            # Try to parse the result as JSON
            try:
                # Find JSON content within the response
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    # Try to find JSON without markdown markers
                    json_match = re.search(r'({.*})', result_text, re.DOTALL)
                    if json_match:
                        json_content = json_match.group(1)
                    else:
                        json_content = result_text
                
                # Parse the JSON
                interpreted_data = json.loads(json_content)
                return interpreted_data
            except Exception as parse_error:
                logger.error(f"Error parsing model output as JSON: {parse_error}")
                logger.debug(f"Raw model output: {result_text}")
                # Return a simplified interpretation that won't fail
                simplified = {
                    "questions": {},
                    "metadata": {"error": "Failed to parse model output"},
                    "summary": "Interpretation failed, using raw OCR data"
                }
                # Put all content in "unassigned" category
                all_text = "\n\n".join([f"{page}: {content}" for page, content in ocr_data.items()])
                simplified["questions"]["unassigned"] = {
                    "question_number": "unassigned",
                    "raw_text": all_text,
                    "equations": [],
                    "figures": [],
                    "calculations": [],
                    "continues_from_previous": False,
                    "continues_to_next": False
                }
                return simplified
                
        except Exception as e:
            logger.error(f"Error during interpretation: {e}")
            # Return a basic structure containing the raw data
            return {
                "questions": {"error": {"question_number": "error", "raw_text": str(ocr_data)}},
                "metadata": {"error": str(e)},
                "summary": "Error occurred during interpretation"
            }
    
    return run_chain

def interpret_ocr_output(ocr_data, output_dir, app_config=None, api_key=None):
    """Interpret OCR output to create a structured representation.
    
    Args:
        ocr_data: Dictionary of OCR output by page
        output_dir: Directory to save output files
        app_config: Optional application configuration dictionary
        api_key: API key for the model provider
    
    Returns:
        Structured interpretation of the OCR data
    """
    # Get configuration from app_config or use default
    config = DefaultConfig.INTERPRETER.copy() if hasattr(DefaultConfig, 'INTERPRETER') else {
        "temperature": 0.1,
        "max_tokens": 4000,
        "model": "gemini-2.5-pro-exp-03-25",
        "provider": "google"
    }
    
    if app_config and "interpreter" in app_config:
        config.update(app_config["interpreter"])
    
    # Save the API call request for audit purposes
    with open(output_dir / "interpreter_request_log.txt", "a") as f:
        f.write(f"\n--- Interpretation Request at {datetime.now().isoformat()} ---\n")
        f.write(f"Interpreting OCR output with {len(ocr_data)} pages\n")
    
    try:
        # Get parameters from config
        temperature = config.get("temperature", 0.1)
        max_tokens = config.get("max_tokens", 4000)
        model_name = config.get("model", "gemini-2.5-pro-exp-03-25")
        provider = config.get("provider", "google")
        
        # Create interpretation chain
        chain = create_interpreter_chain(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            provider=provider
        )
        
        # Run the chain
        logger.info(f"Running interpreter chain with {provider} model {model_name}")
        interpreted_data = chain(ocr_data)
        
        # Save interpretation result
        with open(output_dir / "interpreted_answers.json", "w") as f:
            json.dump(interpreted_data, f, indent=2)
        
        # Also save a human-readable version
        try:
            with open(output_dir / "interpreted_answers_readable.txt", "w") as f:
                # Write summary
                if "summary" in interpreted_data:
                    f.write(f"SUMMARY: {interpreted_data['summary']}\n\n")
                
                # Write each question
                if "questions" in interpreted_data:
                    for q_num, q_data in interpreted_data["questions"].items():
                        f.write(f"QUESTION {q_num}:\n")
                        f.write(f"{'=' * 50}\n")
                        
                        # Note if question continues from previous or to next
                        if q_data.get("continues_from_previous", False):
                            f.write("[CONTINUES FROM PREVIOUS PAGE]\n")
                        
                        # Write raw text
                        if "raw_text" in q_data:
                            f.write(f"{q_data['raw_text']}\n\n")
                        
                        # Write specialized content if available
                        if "equations" in q_data and q_data["equations"]:
                            f.write("EQUATIONS:\n")
                            for i, eq in enumerate(q_data["equations"], 1):
                                f.write(f"{i}. {eq}\n")
                            f.write("\n")
                        
                        if "figures" in q_data and q_data["figures"]:
                            f.write("FIGURES:\n")
                            for i, fig in enumerate(q_data["figures"], 1):
                                f.write(f"{i}. {fig}\n")
                            f.write("\n")
                        
                        if "calculations" in q_data and q_data["calculations"]:
                            f.write("CALCULATIONS:\n")
                            for i, calc in enumerate(q_data["calculations"], 1):
                                f.write(f"{i}. {calc}\n")
                            f.write("\n")
                        
                        if q_data.get("continues_to_next", False):
                            f.write("[CONTINUES TO NEXT PAGE]\n")
                        
                        f.write("\n\n")
        except Exception as e:
            logger.error(f"Error creating readable interpretation: {e}")
        
        return interpreted_data
    except Exception as e:
        logger.error(f"Error during interpretation process: {e}")
        return {"error": str(e)} 