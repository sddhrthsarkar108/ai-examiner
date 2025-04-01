import os
import sys
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

from config import load_app_config, create_default_questions_config, load_questions, DefaultConfig
from utils import setup_logging, validate_pdf_path, create_output_directory, create_evaluation_directory
from ocr import extract_answers_from_pdf
from interpreter import interpret_ocr_output
from evaluation import evaluate_answers, parse_evaluation_results, generate_aggregate_summary
from llm_factory import LLMFactory

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging()

def display_configuration_info(app_config):
    """
    Display information about the current configuration and how to modify it.
    
    Args:
        app_config: The current application configuration
    """
    print("\n" + "="*80)
    print("CONFIGURATION INFORMATION")
    print("="*80)
    print("The application is using the following configuration from environment variables:")
    
    # OCR Configuration
    print("\nOCR Configuration:")
    print(f"  - Provider: {app_config['ocr'].get('provider', 'openai')}")
    print(f"  - Model: {app_config['ocr'].get('model', 'gpt-4o')}")
    print(f"  - Temperature: {app_config['ocr'].get('temperature', 0.0)}")
    print(f"  - Max Tokens: {app_config['ocr'].get('max_tokens', 4000)}")
    print(f"  - Include Examples: {app_config['ocr'].get('include_examples', False)}")
    api_key_status = "Custom key" if app_config['ocr'].get('api_key') else "From environment"
    print(f"  - API Key: {api_key_status}")
    
    # Interpreter Configuration
    print("\nInterpreter Configuration:")
    print(f"  - Provider: {app_config['interpreter'].get('provider', 'google')}")
    print(f"  - Model: {app_config['interpreter'].get('model', 'gemini-2.5-pro-exp-03-25')}")
    print(f"  - Temperature: {app_config['interpreter'].get('temperature', 0.1)}")
    print(f"  - Max Tokens: {app_config['interpreter'].get('max_tokens', 4000)}")
    api_key_status = "Custom key" if app_config['interpreter'].get('api_key') else "From environment"
    print(f"  - API Key: {api_key_status}")
    
    # Evaluation Configuration
    print("\nEvaluation Configuration:")
    print(f"  - Provider: {app_config['evaluation'].get('provider', 'deepseek')}")
    print(f"  - Model: {app_config['evaluation'].get('model', 'deepseek-chat')}")
    print(f"  - Temperature: {app_config['evaluation'].get('temperature', 0.1)}")
    print(f"  - Max Tokens: {app_config['evaluation'].get('max_tokens', 1500)}")
    print(f"  - Number of Evaluations: {app_config['evaluation'].get('num_evaluations', 3)}")
    api_key_status = "Custom key" if app_config['evaluation'].get('api_key') else "From environment"
    print(f"  - API Key: {api_key_status}")
    
    # General Configuration
    print("\nGeneral Configuration:")
    print(f"  - Domain Context: {app_config['general'].get('domain_context', 'machine learning exam')}")
    print(f"  - Output Directory: {app_config['general'].get('output_directory', 'output')}")
    print(f"  - Log Level: {app_config['general'].get('log_level', 'INFO')}")
    
    # Behavior Settings
    print("\nBehavior Settings:")
    print(f"  - Show Config: {app_config['behavior'].get('show_config', False)}")
    print(f"  - Default PDF Path: {app_config['behavior'].get('default_pdf_path', '')}")
    print(f"  - Auto Reuse OCR: {app_config['behavior'].get('auto_reuse_ocr', True)}")
    print(f"  - Auto Reuse Interpreter: {app_config['behavior'].get('auto_reuse_interpreter', True)}")
    
    # API Key Information
    print("\nAPI Key Information:")
    print("  API keys are loaded from environment variables:")
    print("  - OpenAI: OPENAI_API_KEY")
    print("  - Google: GOOGLE_API_KEY")
    print("  - DeepSeek: DEEPSEEK_API_KEY or DEEPINFRA_API_TOKEN")
    
    # Instructions for modifying configuration
    print("\nTo modify these settings, update your .envrc file:")
    print("  For example, to change the OCR model:")
    print("  export OCR_MODEL=gpt-4o")
    print("\nChanges to environment variables will be applied on the next run.")
    print("="*80 + "\n")

def main():
    """
    Main workflow for the automated answer script grading system.
    
    The system follows a three-stage pipeline:
    1. OCR: Extract raw text from handwritten PDFs (responsibility: accurate transcription)
    2. Interpreter: Structure OCR output by questions (responsibility: organization)
    3. Evaluation: Grade the interpreted answers (responsibility: assessment)
    
    Each stage runs only once per PDF by default, with results cached for efficiency.
    Each component can use a different LLM provider based on the configuration.
    """
    try:
        logger.info("Starting automated answer script grading pipeline")
        
        # Load questions
        questions = load_questions()
        
        # Load application configuration from environment variables
        app_config = load_app_config()
        logger.info(f"Application configuration loaded from environment variables")
        
        # Get behavior settings
        behavior = app_config.get("behavior", {})
        show_config = behavior.get("show_config", False)
        default_pdf_path = behavior.get("default_pdf_path", "")
        auto_reuse_ocr = behavior.get("auto_reuse_ocr", True)
        auto_reuse_interpreter = behavior.get("auto_reuse_interpreter", True)
        
        # Display configuration information if enabled
        if show_config:
            display_configuration_info(app_config)
        
        # Get provider information from config
        ocr_config = app_config.get("ocr", DefaultConfig.OCR)
        interpreter_config = app_config.get("interpreter", DefaultConfig.INTERPRETER)
        eval_config = app_config.get("evaluation", DefaultConfig.EVALUATION)
        
        ocr_provider = ocr_config.get("provider", "openai").lower()
        interpreter_provider = interpreter_config.get("provider", "google").lower()
        eval_provider = eval_config.get("provider", "deepseek").lower()
        
        # Get API keys from config if available
        ocr_api_key = ocr_config.get("api_key")
        interpreter_api_key = interpreter_config.get("api_key")
        evaluation_api_key = eval_config.get("api_key")
        
        logger.info(f"Using {ocr_provider} for OCR, {interpreter_provider} for interpretation, and {eval_provider} for evaluation")
        
        # Get PDF path from user or environment
        answer_script_pdf = default_pdf_path
        if not answer_script_pdf:
            answer_script_pdf = input("Enter the path to the answer script PDF: ")
        else:
            logger.info(f"Using default PDF path from environment: {answer_script_pdf}")
        
        try:
            answer_script_pdf = validate_pdf_path(answer_script_pdf)
        except (FileNotFoundError, ValueError) as e:
            logger.error(e)
            sys.exit(1)
            
        # Create main output directory based on PDF name
        pdf_base_name = Path(answer_script_pdf).stem
        base_output_dir = Path("output") / pdf_base_name
        os.makedirs(base_output_dir, exist_ok=True)
        
        #######################
        # STAGE 1: OCR PROCESSING
        #######################
        # OCR Responsibility: Accurate transcription of text with basic formatting markers
        
        # Create OCR directory
        ocr_dir = base_output_dir / "ocr"
        os.makedirs(ocr_dir, exist_ok=True)
        
        # Check if OCR results already exist
        ocr_results_path = ocr_dir / "processed_answers.json"
        run_ocr = True
        
        if ocr_results_path.exists():
            # Use auto_reuse_ocr setting to determine whether to reuse or run again
            if auto_reuse_ocr:
                logger.info("OCR results already exist. Automatically reusing them based on environment settings.")
                # Load existing OCR results
                try:
                    with open(ocr_results_path, 'r') as f:
                        ocr_results = json.load(f)
                    logger.info(f"Loaded OCR results with {len(ocr_results)} pages")
                    run_ocr = False
                except Exception as e:
                    logger.error(f"Error loading existing OCR results: {e}")
                    logger.info("Will run OCR again")
                    run_ocr = True
            else:
                # Auto_reuse_ocr is False, so automatically run OCR again without asking
                logger.info("OCR results exist but AUTO_REUSE_OCR is set to False. Running OCR again...")
                run_ocr = True
        
        # Run OCR if needed
        if run_ocr:
            logger.info("STAGE 1: Extracting text from answer script using OCR...")
            try:
                # Get the API key from config or environment via LLMFactory
                ocr_results = extract_answers_from_pdf(
                    answer_script_pdf, 
                    ocr_dir, 
                    app_config, 
                    ocr_api_key
                )
                logger.info(f"Extracted text from {len(ocr_results)} pages")
            except Exception as e:
                logger.error(f"Failed to extract text: {e}")
                sys.exit(1)
        
        #######################
        # STAGE 2: INTERPRETER PROCESSING
        #######################
        # Interpreter Responsibility: Organize content by question and structure the data
        
        # Create interpreter directory
        interpreter_dir = base_output_dir / "interpreter"
        os.makedirs(interpreter_dir, exist_ok=True)
        
        # Check if interpreted results already exist
        interpreted_results_path = interpreter_dir / "interpreted_answers.json"
        run_interpreter = True
        
        if interpreted_results_path.exists():
            # Use auto_reuse_interpreter setting to determine whether to reuse or run again
            if auto_reuse_interpreter:
                logger.info("Interpreted results already exist. Automatically reusing them based on environment settings.")
                # Load existing interpreted results
                try:
                    with open(interpreted_results_path, 'r') as f:
                        interpreted_answers = json.load(f)
                    logger.info(f"Loaded interpreted answers with {len(interpreted_answers.get('questions', {}))} questions")
                    run_interpreter = False
                except Exception as e:
                    logger.error(f"Error loading existing interpreted results: {e}")
                    logger.info("Will run interpreter again")
                    run_interpreter = True
            else:
                # Auto_reuse_interpreter is False, so automatically run interpreter again without asking
                logger.info("Interpreter results exist but AUTO_REUSE_INTERPRETER is set to False. Running interpreter again...")
                run_interpreter = True
        
        # Run interpreter if needed
        if run_interpreter:
            logger.info("STAGE 2: Interpreting OCR results to organize content by questions...")
            try:
                # Get the API key from config or environment via LLMFactory
                interpreted_answers = interpret_ocr_output(
                    ocr_results, 
                    interpreter_dir, 
                    app_config,
                    interpreter_api_key
                )
                logger.info(f"Interpreted answers into {len(interpreted_answers.get('questions', {}))} questions")
                
                # Save a human-readable version of the interpretation 
                with open(interpreter_dir / "interpreted_answers_readable.txt", "w") as f:
                    f.write(f"# INTERPRETED ANSWER SUMMARY\n\n")
                    f.write(f"Total Questions: {interpreted_answers.get('metadata', {}).get('total_questions', 'Unknown')}\n")
                    f.write(f"Pages Processed: {interpreted_answers.get('metadata', {}).get('pages_processed', 'Unknown')}\n\n")
                    
                    for q_num, q_data in interpreted_answers.get("questions", {}).items():
                        f.write(f"## QUESTION {q_num}\n\n")
                        f.write("- Raw Text:\n\n```\n")
                        f.write(q_data.get("raw_text", "No raw text available"))
                        f.write("\n```\n\n")
                        
                        # Write special elements
                        if q_data.get("equations", []):
                            f.write("- Equations:\n")
                            for i, eq in enumerate(q_data.get("equations", []), 1):
                                f.write(f"  {i}. {eq}\n")
                            f.write("\n")
                            
                        if q_data.get("figures", []):
                            f.write("- Figures:\n")
                            for i, fig in enumerate(q_data.get("figures", []), 1):
                                f.write(f"  {i}. {fig}\n")
                            f.write("\n")
                            
                        if q_data.get("calculations", []):
                            f.write("- Calculations:\n")
                            for i, calc in enumerate(q_data.get("calculations", []), 1):
                                f.write(f"  {i}. {calc}\n")
                            f.write("\n")
                            
                        # Write continuation flags
                        if q_data.get("continues_from_previous", False):
                            f.write("- Continues from previous: Yes\n")
                        if q_data.get("continues_to_next", False):
                            f.write("- Continues to next: Yes\n")
                        f.write("\n---\n\n")
                
            except Exception as e:
                logger.error(f"Failed to interpret OCR results: {e}")
                sys.exit(1)
        
        #######################
        # PREPARE DATA FOR EVALUATION
        #######################
        # Create a well-formatted text version of the interpreted answers for evaluation
        
        student_answers_text = ""
        
        # Try to create a well-formatted text from the interpreted answers
        for q_num, q_data in interpreted_answers.get("questions", {}).items():
            if q_num.lower() == "unassigned" or q_num.lower() == "error":
                # Skip unassigned content or errors
                continue
                
            student_answers_text += f"[Q{q_num}]\n"
            
            if q_data.get("continues_from_previous", False):
                student_answers_text += "[CONTINUES_FROM_PREVIOUS]\n"
            
            student_answers_text += q_data.get("raw_text", "") + "\n"
            
            # Add specialized content if not already in raw_text
            for eq in q_data.get("equations", []):
                if eq not in q_data.get("raw_text", ""):
                    student_answers_text += f"[EQUATION]{eq}[/EQUATION]\n"
            
            for fig in q_data.get("figures", []):
                if fig not in q_data.get("raw_text", ""):
                    student_answers_text += f"[FIGURE]{fig}[/FIGURE]\n"
            
            for calc in q_data.get("calculations", []):
                if calc not in q_data.get("raw_text", ""):
                    student_answers_text += f"[CALCULATION]{calc}[/CALCULATION]\n"
            
            if q_data.get("continues_to_next", False):
                student_answers_text += "[CONTINUES_TO_NEXT]\n"
            
            student_answers_text += "\n"
        
        # If the formatted answers are empty, fall back to OCR results
        if not student_answers_text.strip():
            logger.warning("Couldn't create structured answers from interpretation, falling back to OCR results")
            student_answers_text = "\n\n".join([f"{page}: {content}" for page, content in ocr_results.items()])
        
        # Save the evaluation-ready answers
        with open(interpreter_dir / "evaluation_ready_answers.txt", "w") as f:
            f.write(student_answers_text)
        
        #######################
        # STAGE 3: EVALUATION PROCESSING
        #######################
        # Evaluation Responsibility: Assess content quality and correctness
        
        # Generate a unique directory for this evaluation run
        # Find the highest run number
        existing_runs = [d for d in os.listdir(base_output_dir) 
                        if os.path.isdir(os.path.join(base_output_dir, d)) and d.startswith("run_")]
        
        if existing_runs:
            # Extract run numbers and find the highest
            run_numbers = [int(run.split("_")[1]) for run in existing_runs]
            next_run = max(run_numbers) + 1
        else:
            next_run = 1
        
        # Create run directory for this evaluation batch
        output_dir = base_output_dir / f"run_{next_run}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # Get evaluation configuration from app config
        eval_config = app_config.get("evaluation", DefaultConfig.EVALUATION.copy())
        num_evaluations = eval_config.get("num_evaluations", DefaultConfig.EVALUATION["num_evaluations"])
        
        # Check if multi-provider evaluation is enabled
        multi_provider_eval = eval_config.get("multi_provider_evaluation", False)
        providers_config = eval_config.get("providers_config", {})
        
        # Track which providers were used for the summary
        used_providers = []
        
        # Run multiple evaluations
        logger.info(f"STAGE 3: Running evaluation to assess answer quality ({num_evaluations} runs)...")
        if multi_provider_eval:
            logger.info(f"Using multiple providers for evaluation (multi-provider mode enabled)")
            print("\n" + "="*80)
            print(f"MULTI-PROVIDER EVALUATION MODE ENABLED")
            print(f"Using different LLMs for each evaluation run:")
            for run_num in range(1, num_evaluations + 1):
                run_key = f"run_{run_num}"
                if run_key in providers_config:
                    provider = providers_config[run_key].get("provider", "deepseek")
                    model = providers_config[run_key].get("model", "deepseek-chat")
                    print(f"→ Run {run_num}: {provider.upper()} - {model}")
            print("="*80)
        
        all_summaries = []
        for eval_run in range(1, num_evaluations + 1):
            # Create directory for this evaluation run
            eval_output_dir = create_evaluation_directory(output_dir, eval_run)
            logger.info(f"Created evaluation directory: {eval_output_dir}")
            
            # Get run-specific configuration if multi-provider evaluation is enabled
            run_config = None
            if multi_provider_eval:
                run_key = f"run_{eval_run}"
                if run_key in providers_config:
                    run_config = providers_config.get(run_key)
                    provider_name = run_config.get("provider", eval_config.get("provider", "deepseek"))
                    model_name = run_config.get("model", eval_config.get("model", "deepseek-chat"))
                    
                    # Enhanced logging with clear demarcation
                    logger.info(f"{'*'*50}")
                    logger.info(f"EVALUATION RUN {eval_run}: Using {provider_name.upper()} with model {model_name}")
                    logger.info(f"{'*'*50}")
                    
                    # Print to console for clear visibility
                    print(f"\n{'='*80}")
                    print(f"STARTING EVALUATION RUN {eval_run} WITH {provider_name.upper()} ({model_name})")
                    print(f"{'='*80}")
                else:
                    provider_name = eval_config.get("provider", "deepseek")
                    model_name = eval_config.get("model", "deepseek-chat")
                    logger.warning(f"No specific configuration found for {run_key}, using default provider '{provider_name}'")
                    
                    # Enhanced logging for fallback
                    print(f"\n{'='*80}")
                    print(f"STARTING EVALUATION RUN {eval_run} WITH DEFAULT PROVIDER {provider_name.upper()} ({model_name})")
                    print(f"{'='*80}")
            else:
                provider_name = eval_config.get("provider", "deepseek")
                model_name = eval_config.get("model", "deepseek-chat")
                
                # Standard logging for single-provider mode
                print(f"\n{'='*80}")
                print(f"STARTING EVALUATION RUN {eval_run} WITH {provider_name.upper()} ({model_name})")
                print(f"{'='*80}")
            
            # Track which provider is used for this run
            used_providers.append(provider_name)
            
            # Evaluate answers
            logger.info(f"Running evaluation {eval_run}/{num_evaluations}...")
            try:
                logger.info(f"Using domain context for evaluation: {app_config['general'].get('domain_context', DefaultConfig.GENERAL['domain_context'])}")
                
                # Get the API key from config or environment via LLMFactory
                evaluation_result = evaluate_answers(
                    questions, 
                    student_answers_text, 
                    eval_output_dir, 
                    app_config, 
                    evaluation_api_key,
                    run_config
                )
                # Enhanced completion logging
                logger.info(f"{'✓'*50}")
                logger.info(f"EVALUATION RUN {eval_run} COMPLETED SUCCESSFULLY WITH {provider_name.upper()}")
                logger.info(f"{'✓'*50}")
            except Exception as e:
                logger.error(f"{'!'*50}")
                logger.error(f"EVALUATION RUN {eval_run} FAILED WITH {provider_name.upper()}: {e}")
                logger.error(f"{'!'*50}")
                continue
                
            # Generate summary
            logger.info(f"Generating evaluation summary for run {eval_run} (provider: {provider_name})...")
            df_summary = parse_evaluation_results(evaluation_result, eval_output_dir, questions)
            all_summaries.append(df_summary)
            
            # Save summary to CSV
            csv_path = eval_output_dir / "evaluation_summary.csv"
            df_summary.to_csv(csv_path, index=False)
            logger.info(f"Summary saved to {csv_path}")
            
            # Print summary for this evaluation with enhanced header
            print(f"\n{'#'*80}")
            print(f"EVALUATION {eval_run} RESULTS - {provider_name.upper()} ({model_name})")
            print(f"{'#'*80}")
            print(df_summary[["Question No.", "Max Marks", "Marks Obtained", "Justification"]].to_string(index=False))
            
            # Add separator for readability
            print(f"\n{'.'*80}\n")
        
        #######################
        # AGGREGATE RESULTS
        #######################
        # Generate aggregate summary across all evaluations
        if all_summaries:
            print("\n" + "="*80)
            print("Generating aggregate summary with voting-based system...")
            print("="*80)
            
            aggregate_df = generate_aggregate_summary(all_summaries, output_dir)
            
            # Print a simplified version of the aggregate summary
            print("\nAggregate Evaluation Summary (Voting-Based System):")
            
            # Create a simplified view with key columns
            display_columns = ["Question No.", "Max Marks"]
            
            # Add run scores
            num_runs = len(all_summaries)
            for i in range(num_runs):
                display_columns.append(f"Run {i+1} Score")
            
            # Add aggregate columns
            display_columns.extend(["Voting Score", "Average Score"])
            
            # Display the simplified table
            print(aggregate_df[display_columns].to_string(index=False))
            
            # Print explanation of the voting system
            print("\nVoting System Details:")
            print("- Voting Score: Most common score across runs (rounded to nearest 0.5)")
            print("- In case of ties, the highest score is selected")
            print("- Average Score: Simple mean of all run scores")
            
            print(f"\nDetailed results with justifications saved to:")
            print(f"- CSV: {output_dir / 'aggregate_evaluation_summary.csv'}")
            print(f"- HTML: {output_dir / 'aggregate_evaluation_summary.html'}")
        
        # Provide summary of pipeline execution and output locations
        print("\n" + "="*80)
        print("AUTOMATED ANSWER SCRIPT PIPELINE COMPLETED")
        print("="*80)
        print(f"Pipeline stages with clear responsibility boundaries:")
        print(f"1. OCR (using {ocr_provider}): Extracted raw text with formatting markers")
        print(f"2. Interpreter (using {interpreter_provider}): Organized content by questions")
        
        # Show evaluation providers used
        if multi_provider_eval and used_providers:
            print(f"3. Evaluation: Assessed answer quality using MULTIPLE PROVIDERS")
            print("\n   EVALUATION PROVIDERS USED:")
            print("   " + "-"*50)
            print("   | Run |   Provider   |        Model        |")
            print("   " + "-"*50)
            
            for i, provider in enumerate(used_providers):
                run_key = f"run_{i+1}"
                if run_key in providers_config:
                    provider_cfg = providers_config[run_key]
                    model = provider_cfg.get("model", "unknown")
                    temp = provider_cfg.get("temperature", "N/A")
                else:
                    model = eval_config.get("model", "unknown")
                    temp = eval_config.get("temperature", "N/A")
                
                # Format provider name with capitalization and padding
                provider_disp = f"{provider.upper():^13}"
                
                # Format model name with padding
                model_disp = f"{model:^20}"
                
                # Print the table row
                print(f"   |  {i+1}  | {provider_disp} | {model_disp} |")
            
            print("   " + "-"*50)
            print("\n   VOTING-BASED AGGREGATION:")
            print("   → Final scores determined by voting across all evaluation runs")
            print("   → Run 2 used GOOGLE provider (Gemini) while runs 1 & 3 used DEEPSEEK")
            print("   → This multi-provider approach increases evaluation robustness")
        else:
            print(f"3. Evaluation (using {eval_provider}): Assessed answer quality and correctness")
        
        print("\nOutput locations:")
        print(f"- OCR results: {ocr_dir}")
        print(f"- Interpreted results: {interpreter_dir}")
        print(f"- Evaluation results: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
