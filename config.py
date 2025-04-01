import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configure module-level logger
logger = logging.getLogger(__name__)


# Default configurations
class DefaultConfig:
    """Default configuration values for the application."""

    # OCR configuration
    OCR = {
        "max_tokens": 4000,
        "include_examples": False,
        "detail_level": "high",
        "temperature": 0.0,  # Low temperature for more consistent extraction
        "model": "gpt-4o",  # OpenAI model with vision capabilities
        "provider": "openai",  # Model provider (default: OpenAI for vision capabilities)
        "api_key": None,  # API key (default: None, will be loaded from environment)
        "non_ocr_mode": False,  # Whether to use non-OCR mode (read existing text files)
        "extracted_texts_dir": "extracted_texts",  # Directory containing extracted text files
    }

    # Interpreter configuration
    INTERPRETER = {
        "max_tokens": 4000,
        "temperature": 0.1,  # Slightly higher for interpretation flexibility
        "model": "gemini-2.5-pro-exp-03-25",  # Google's model
        "provider": "google",  # Model provider (default: Google for Gemini model)
        "api_key": None,  # API key (default: None, will be loaded from environment)
    }

    # Evaluation configuration
    EVALUATION = {
        "num_evaluations": 3,
        "max_tokens": 1500,
        "temperature": 0.1,  # Slightly higher temperature for evaluation flexibility
        "model": "deepseek-chat",  # Default model for evaluation
        "provider": "deepseek",  # Model provider (default: DeepSeek for evaluation)
        "api_key": None,  # API key (default: None, will be loaded from environment)
        "multi_provider_evaluation": False,  # Whether to use multiple providers for evaluation
        "providers_config": {
            # Default providers for each evaluation run when multi_provider_evaluation is enabled
            "run_1": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 1500,
            },
            "run_2": {
                "provider": "google",
                "model": "gemini-2.5-pro-exp-03-25",
                "temperature": 0.1,
                "max_tokens": 4000,
            },
            "run_3": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 1500,
            },
        },
    }

    # General application configuration
    GENERAL = {
        "output_directory": "output",
        "log_level": "INFO",
        "domain_context": "machine learning exam",
    }

    # Complete application configuration
    APP = {
        "ocr": OCR.copy(),
        "interpreter": INTERPRETER.copy(),
        "evaluation": EVALUATION.copy(),
        "general": GENERAL.copy(),
    }

    # App behavior settings
    BEHAVIOR = {
        "show_config": False,  # Whether to show configuration on startup
        "default_pdf_path": "",  # Default PDF path to process, if empty will prompt user
        "auto_reuse_ocr": True,  # Automatically reuse existing OCR results if available
        "auto_reuse_interpreter": True,  # Automatically reuse existing interpreter results if available
        "process_all_files": False,  # Whether to process all files in extracted_texts when in non-OCR mode
        "skip_existing_outputs": True,  # Whether to skip processing when output directory already exists
    }


def parse_bool(value: str) -> bool:
    """
    Parse a string value to boolean.

    Args:
        value: String value to parse

    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value

    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "n", "0"):
        return False
    else:
        raise ValueError(f"Cannot parse '{value}' to boolean")


def parse_int(value: str, default: Optional[int] = None) -> int:
    """
    Parse a string value to integer.

    Args:
        value: String value to parse
        default: Default value if parsing fails

    Returns:
        Integer value or default
    """
    if value is None or value == "":
        return default

    try:
        return int(value)
    except ValueError:
        if default is not None:
            logger.warning(
                f"Failed to parse '{value}' to int, using default: {default}"
            )
            return default
        else:
            raise


def parse_float(value: str, default: Optional[float] = None) -> float:
    """
    Parse a string value to float.

    Args:
        value: String value to parse
        default: Default value if parsing fails

    Returns:
        Float value or default
    """
    if value is None or value == "":
        return default

    try:
        return float(value)
    except ValueError:
        if default is not None:
            logger.warning(
                f"Failed to parse '{value}' to float, using default: {default}"
            )
            return default
        else:
            raise


def load_from_env(prefix: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load configuration from environment variables based on prefix.

    Args:
        prefix: Prefix for environment variables (e.g., "OCR_")
        default_config: Default configuration to use for missing values

    Returns:
        Configuration dictionary
    """
    config = {}

    for key, default_value in default_config.items():
        env_key = f"{prefix}_{key}".upper()
        env_value = os.environ.get(env_key)

        if env_value is not None:
            # Parse value based on the type of the default value
            if isinstance(default_value, bool):
                config[key] = parse_bool(env_value)
            elif isinstance(default_value, int):
                config[key] = parse_int(env_value, default_value)
            elif isinstance(default_value, float):
                config[key] = parse_float(env_value, default_value)
            # Handle nested dictionaries like providers_config separately
            elif isinstance(default_value, dict) and key == "providers_config":
                # Just use the default value for providers_config
                config[key] = default_value
            else:
                config[key] = env_value
        else:
            # Use default value if no environment variable exists
            config[key] = default_value

    return config


def load_app_config() -> Dict[str, Any]:
    """
    Load application configuration from environment variables.
    Falls back to default values if environment variables are not set.

    Returns:
        dict: Application configuration dictionary
    """
    logger.info("Loading configuration from environment variables")

    # Load configuration from environment variables
    config = {
        "ocr": load_from_env("OCR", DefaultConfig.OCR),
        "interpreter": load_from_env("INTERPRETER", DefaultConfig.INTERPRETER),
        "evaluation": load_from_env("EVALUATION", DefaultConfig.EVALUATION),
        "general": load_from_env("GENERAL", DefaultConfig.GENERAL),
    }

    # Load behavior settings
    behavior = {}
    for key, default_value in DefaultConfig.BEHAVIOR.items():
        env_key = key.upper()
        env_value = os.environ.get(env_key)

        if env_value is not None:
            if isinstance(default_value, bool):
                behavior[key] = parse_bool(env_value)
            else:
                behavior[key] = env_value
        else:
            behavior[key] = default_value

    config["behavior"] = behavior

    # Validate the configuration
    if validate_config(config):
        logger.info("Configuration validated successfully")
    else:
        logger.warning("Configuration validation failed, using default values")
        config = DefaultConfig.APP.copy()
        config["behavior"] = DefaultConfig.BEHAVIOR.copy()

    return config


def validate_config(config):
    """Validate the structure and values of the configuration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if valid, False otherwise
    """
    # Check for required top-level sections
    required_sections = ["ocr", "evaluation", "general"]
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False

    # Validate OCR section
    ocr_config = config.get("ocr", {})
    if not isinstance(ocr_config.get("max_tokens"), int):
        logger.error("OCR max_tokens must be an integer")
        return False

    if not isinstance(ocr_config.get("include_examples"), bool):
        logger.error("OCR include_examples must be a boolean")
        return False

    if "temperature" in ocr_config and not isinstance(
        ocr_config["temperature"], (int, float)
    ):
        logger.error("OCR temperature must be a number")
        return False

    if "model" in ocr_config and not isinstance(ocr_config["model"], str):
        logger.error("OCR model must be a string")
        return False

    if "provider" in ocr_config and not isinstance(ocr_config["provider"], str):
        logger.error("OCR provider must be a string")
        return False

    if (
        "api_key" in ocr_config
        and ocr_config["api_key"] is not None
        and not isinstance(ocr_config["api_key"], str)
    ):
        logger.error("OCR api_key must be a string or null")
        return False

    # Validate evaluation section
    eval_config = config.get("evaluation", {})
    if "num_evaluations" in eval_config and not isinstance(
        eval_config["num_evaluations"], int
    ):
        logger.error("Evaluation num_evaluations must be an integer")
        return False

    if "max_tokens" in eval_config and not isinstance(eval_config["max_tokens"], int):
        logger.error("Evaluation max_tokens must be an integer")
        return False

    if "temperature" in eval_config and not isinstance(
        eval_config["temperature"], (int, float)
    ):
        logger.error("Evaluation temperature must be a number")
        return False

    if "model" in eval_config and not isinstance(eval_config["model"], str):
        logger.error("Evaluation model must be a string")
        return False

    if "provider" in eval_config and not isinstance(eval_config["provider"], str):
        logger.error("Evaluation provider must be a string")
        return False

    if (
        "api_key" in eval_config
        and eval_config["api_key"] is not None
        and not isinstance(eval_config["api_key"], str)
    ):
        logger.error("Evaluation api_key must be a string or null")
        return False

    # Validate multi-provider evaluation settings if enabled
    if "multi_provider_evaluation" in eval_config:
        if not isinstance(eval_config["multi_provider_evaluation"], bool):
            logger.error("Evaluation multi_provider_evaluation must be a boolean")
            return False

        if (
            eval_config["multi_provider_evaluation"]
            and "providers_config" in eval_config
        ):
            if not isinstance(eval_config["providers_config"], dict):
                logger.error("Evaluation providers_config must be a dictionary")
                return False

            # Validate each run configuration
            for run_key, run_config in eval_config["providers_config"].items():
                if not isinstance(run_config, dict):
                    logger.error(
                        f"Evaluation providers_config.{run_key} must be a dictionary"
                    )
                    return False

                # Validate required fields in run configuration
                if "provider" in run_config and not isinstance(
                    run_config["provider"], str
                ):
                    logger.error(
                        f"Evaluation providers_config.{run_key}.provider must be a string"
                    )
                    return False

                if "model" in run_config and not isinstance(run_config["model"], str):
                    logger.error(
                        f"Evaluation providers_config.{run_key}.model must be a string"
                    )
                    return False

    # Validate interpreter section if present
    if "interpreter" in config:
        interpreter_config = config["interpreter"]
        if "max_tokens" in interpreter_config and not isinstance(
            interpreter_config["max_tokens"], int
        ):
            logger.error("Interpreter max_tokens must be an integer")
            return False

        if "temperature" in interpreter_config and not isinstance(
            interpreter_config["temperature"], (int, float)
        ):
            logger.error("Interpreter temperature must be a number")
            return False

        if "model" in interpreter_config and not isinstance(
            interpreter_config["model"], str
        ):
            logger.error("Interpreter model must be a string")
            return False

        if "provider" in interpreter_config and not isinstance(
            interpreter_config["provider"], str
        ):
            logger.error("Interpreter provider must be a string")
            return False

        if (
            "api_key" in interpreter_config
            and interpreter_config["api_key"] is not None
            and not isinstance(interpreter_config["api_key"], str)
        ):
            logger.error("Interpreter api_key must be a string or null")
            return False

    # Validate general section
    general_config = config.get("general", {})
    required_general = ["output_directory", "log_level", "domain_context"]
    for field in required_general:
        if field not in general_config:
            logger.error(f"Missing required field in general config: {field}")
            return False

    return True


def load_questions():
    """Load questions from the questions configuration file.

    Returns:
        dict: Dictionary of questions and their maximum marks

    Raises:
        FileNotFoundError: If questions file is not found
        json.JSONDecodeError: If questions file is not valid JSON
    """
    questions_config_path = Path(__file__).parent / "config" / "questions.json"

    if not questions_config_path.exists():
        logger.error(
            f"Questions configuration file not found at {questions_config_path}"
        )
        raise FileNotFoundError(
            f"Questions configuration file not found at {questions_config_path}"
        )

    try:
        with open(questions_config_path, "r") as f:
            questions = json.load(f)
        logger.info(f"Successfully loaded questions from {questions_config_path}")
        return questions
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing questions configuration file: {str(e)}")
        raise json.JSONDecodeError(
            f"Error parsing questions configuration file: {str(e)}", e.doc, e.pos
        )
    except Exception as e:
        logger.error(f"Error loading questions configuration: {str(e)}")
        raise


def create_default_questions_config():
    """Create default questions configuration file if it doesn't exist.

    Returns:
        dict: The questions configuration (loaded or created)
    """
    questions_config_path = Path(__file__).parent / "config" / "questions.json"

    if not questions_config_path.exists():
        # Create a default questions config
        default_questions = {
            "1": [
                "Explain the differences between supervised and unsupervised learning with examples.",
                10,
            ],
            "2": [
                "Describe how gradient descent works for optimizing a neural network.",
                15,
            ],
            "3": [
                "Explain the bias-variance tradeoff and its significance in machine learning.",
                10,
            ],
            "4": [
                "Define precision and recall, and explain their importance in classification tasks.",
                10,
            ],
            "5": [
                "Explain how regularization helps prevent overfitting in machine learning models.",
                10,
            ],
        }

        ensure_config_directory_exists()

        try:
            with open(questions_config_path, "w") as f:
                json.dump(default_questions, f, indent=2)
            logger.info(
                f"Created default questions configuration at {questions_config_path}"
            )
        except Exception as e:
            logger.error(f"Failed to create default questions configuration: {e}")

    return load_questions()


def ensure_config_directory_exists():
    """Ensure the config directory exists."""
    config_dir = Path(__file__).parent / "config"
    if not config_dir.exists():
        os.makedirs(config_dir, exist_ok=True)
        logger.info(f"Created configuration directory: {config_dir}")
