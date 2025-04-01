import logging
import re
import time
import pandas as pd
from datetime import datetime

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage

from config import DefaultConfig
from llm_factory import LLMFactory
from prompts import get_evaluation_system_prompt, get_evaluation_user_prompt

logger = logging.getLogger(__name__)

def create_evaluation_chain(model_name="deepseek-chat", temperature=0.1, max_tokens=1500, api_key=None, provider="deepseek"):
    """Create a chain for evaluation processing.
    
    Args:
        model_name: The name of the model to use
        temperature: Temperature setting for model output randomness
        max_tokens: Maximum tokens for model response
        api_key: API key for the model provider
        provider: The model provider to use (default: deepseek)
        
    Returns:
        A function that processes evaluation requests
    """
    # Use the LLM factory to create the evaluation chain
    return LLMFactory.create_chain_for_evaluation(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )

def evaluate_answers(questions, answers, eval_output_dir, app_config=None, api_key=None, run_config=None):
    """Evaluate answers based on clarity, completeness, accuracy, examples, and presentation.
    
    Args:
        questions: Dictionary of questions and maximum marks
        answers: Student answers to evaluate
        eval_output_dir: Directory to save evaluation results
        app_config: Application configuration dictionary
        api_key: API key for the model provider (e.g., DeepSeek API key)
        run_config: Optional configuration specific to this evaluation run
    """
    # Get domain context from app_config or use default
    domain_context = "machine learning exam"
    if app_config and "general" in app_config:
        domain_context = app_config["general"].get("domain_context", "machine learning exam")
    
    # Get evaluation configuration from app_config or use default
    eval_config = DefaultConfig.EVALUATION.copy()
    if app_config and "evaluation" in app_config:
        eval_config.update(app_config["evaluation"])
    
    # Override with run-specific configuration if provided
    if run_config:
        for key, value in run_config.items():
            eval_config[key] = value
    
    # Construct a table of questions and marks for the prompt
    questions_table = "\n".join([f"Question {q_no}: {question_text} [{max_marks} marks]" 
                               for q_no, (question_text, max_marks) in questions.items()])
    
    # Create a dynamic question list for the score table template with justification
    question_score_template = "\n".join([f"{q_no},{max_marks},your_score_for_q{q_no},\"Brief justification for question {q_no} score\"" 
                                      for q_no, (_, max_marks) in questions.items()])
    
    # Get prompts from centralized prompts module
    system_prompt = get_evaluation_system_prompt(domain_context)
    user_prompt = get_evaluation_user_prompt(domain_context, questions_table, answers, question_score_template)

    # Get evaluation parameters from config
    temperature = eval_config.get("temperature", 0.1)
    max_tokens = eval_config.get("max_tokens", 1500)
    model_name = eval_config.get("model", "deepseek-chat")
    provider = eval_config.get("provider", "deepseek")

    # Save API request for audit
    with open(eval_output_dir / "api_request_log.txt", "w") as f:
        f.write(f"\n--- Evaluation Request at {datetime.now().isoformat()} ---\n")
        # Don't log the full prompt which contains the answers, just log that we made the request
        f.write(f"Sent evaluation request to {provider} {model_name} API with temperature={temperature}, max_tokens={max_tokens}\n")
    
    try:
        # Add rate limiting
        time.sleep(1)  # 1 second delay between calls
        
        # Create LangChain chain for evaluation
        chain = create_evaluation_chain(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            provider=provider
        )
        
        # Run the chain
        evaluation_result = chain({
            "system": system_prompt,
            "user": user_prompt
        })
        
        # Save result
        with open(eval_output_dir / "evaluation_result.txt", "w") as f:
            f.write(evaluation_result)
            
        return evaluation_result
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

def parse_evaluation_results(evaluation_result, eval_output_dir, questions):
    """Extracts question-wise marks from evaluation results using a delimiter-based approach."""
    # First save the raw evaluation result
    with open(eval_output_dir / "evaluation_result.txt", "w") as f:
        f.write(evaluation_result)
    
    # Extract the table between delimiters (more flexible pattern to handle various whitespace)
    # This pattern is more forgiving of space/newline variations around the SCORE_TABLE tags
    table_match = re.search(r'<\s*SCORE_TABLE\s*>\s*(.*?)\s*<\s*/\s*SCORE_TABLE\s*>', 
                           evaluation_result, re.DOTALL | re.IGNORECASE)
    
    if not table_match:
        # Try alternative patterns if the standard one fails
        alternative_patterns = [
            r'SCORE\s+TABLE:?\s*\n(.*?)\n\s*(\n|$)',  # "SCORE TABLE:" followed by content
            r'score\s+table:?\s*\n(.*?)\n\s*(\n|$)',  # Case-insensitive version
            r'table:?\s*\n(.*?)\n\s*(\n|$)'           # Just "TABLE:" followed by content
        ]
        
        for pattern in alternative_patterns:
            alt_match = re.search(pattern, evaluation_result, re.DOTALL)
            if alt_match:
                logger.warning("Using alternative pattern to extract score table")
                table_match = alt_match
                break
    
    if not table_match:
        logger.error("Could not find score table in the evaluation result")
        # Create empty dataframe with zeros as fallback
        summary = []
        for q_no, (question_text, max_marks) in questions.items():
            logger.warning(f"Could not extract score for question {q_no}, assuming zero")
            summary.append([q_no, question_text, max_marks, 0, "Score not provided"])
    else:
        # Parse the CSV data
        table_content = table_match.group(1).strip()
        
        # Clean up the table content - normalize newlines, remove extra whitespace
        table_content = re.sub(r'\r\n?', '\n', table_content)  # Normalize newlines
        table_content = re.sub(r'\n\s*\n', '\n', table_content)  # Remove empty lines
        table_content = re.sub(r'^\s+|\s+$', '', table_content, flags=re.MULTILINE)  # Trim each line
        
        lines = table_content.split('\n')
        
        # Process header line to identify columns
        header_line = lines[0].lower() if lines else ""
        has_justification = "justification" in header_line
        
        # Build a list of expected question numbers from the questions dictionary
        expected_questions = set(questions.keys())
        processed_questions = set()
        
        # Skip header line
        summary = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            try:
                # More robust CSV parsing using regex with special handling for quoted fields
                # This handles quotes more carefully
                pattern = r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)'
                parts = re.split(pattern, line)
                
                # Clean up the parts (remove quotes, extra whitespace, and non-printable chars)
                parts = [re.sub(r'^\s*["\']|["\']\s*$', '', p).strip() for p in parts]
                
                # Handle incomplete rows more gracefully
                if len(parts) >= 1:  # As long as we have a question number, we can process
                    # Extract and clean question number - remove any non-digit characters if needed
                    q_no_raw = parts[0].strip()
                    q_no_match = re.search(r'(\d+)', q_no_raw)
                    if q_no_match:
                        q_no = q_no_match.group(1)
                    else:
                        q_no = q_no_raw
                    
                    processed_questions.add(q_no)
                    
                    try:
                        # Get default values from questions dictionary
                        question_text, default_max_marks = questions.get(q_no, ["Unknown question", 0])
                        
                        # Try to extract max_marks, using default if not available
                        if len(parts) >= 2 and parts[1].strip():
                            try:
                                # Handle case where commas are used as decimal separators
                                max_marks_str = parts[1].replace(',', '.')
                                max_marks = float(max_marks_str)
                            except ValueError:
                                logger.warning(f"Invalid max marks for question {q_no}, using default: {default_max_marks}")
                                max_marks = default_max_marks
                        else:
                            max_marks = default_max_marks
                        
                        # Try to extract score, defaulting to 0 if not available
                        if len(parts) >= 3 and parts[2].strip():
                            try:
                                # Handle case where commas are used as decimal separators
                                score_str = parts[2].replace(',', '.')
                                score = float(score_str)
                            except ValueError:
                                logger.warning(f"Invalid score for question {q_no}, defaulting to 0")
                                score = 0.0
                        else:
                            logger.warning(f"Missing score for question {q_no}, defaulting to 0")
                            score = 0.0
                        
                        # Extract justification if available, or use default
                        if len(parts) >= 4 and has_justification:
                            justification = parts[3].strip()
                            if not justification:
                                justification = "No justification provided"
                        else:
                            justification = "No justification provided"
                        
                        # Validate score doesn't exceed max marks
                        if score > max_marks:
                            logger.warning(f"Score for question {q_no} ({score}) exceeds max marks ({max_marks}), capping at max")
                            score = max_marks
                        
                        summary.append([q_no, question_text, max_marks, score, justification])
                    except ValueError as e:
                        logger.warning(f"Error parsing score for question {q_no}: {e}")
                        # Use values from our questions dictionary as fallback
                        question_text, max_marks = questions.get(q_no, ["Unknown question", 0])
                        summary.append([q_no, question_text, max_marks, 0, "Error parsing score"])
                else:
                    logger.warning(f"Malformed line in score table (no question number): {line}")
            except Exception as e:
                logger.warning(f"Error processing line in score table: {line}, Error: {e}")
                # Try to extract question number for fallback
                q_no_match = re.match(r'(\d+)', line.strip())
                if q_no_match:
                    q_no = q_no_match.group(1)
                    processed_questions.add(q_no)
                    question_text, max_marks = questions.get(q_no, ["Unknown question", 0])
                    summary.append([q_no, question_text, max_marks, 0, "Error parsing score line"])
        
        # Add any missing questions that weren't in the table
        missing_questions = expected_questions - processed_questions
        for q_no in missing_questions:
            logger.warning(f"Question {q_no} missing from score table, adding with zero score")
            question_text, max_marks = questions.get(q_no, ["Unknown question", 0])
            summary.append([q_no, question_text, max_marks, 0, "Missing from score table"])
    
    # Create dataframe with justification column
    df = pd.DataFrame(summary, columns=["Question No.", "Question", "Max Marks", "Marks Obtained", "Justification"])
    
    # Sort by question number to ensure consistent order
    # Convert question numbers to integers for sorting, handling non-numeric values
    def convert_to_int(q_no):
        try:
            return int(q_no)
        except ValueError:
            return float('inf')  # Place non-numeric at the end
    
    df = df.sort_values(by="Question No.", key=lambda x: x.map(convert_to_int))
    
    # Calculate total
    total_marks = df["Max Marks"].sum()
    obtained_marks = df["Marks Obtained"].sum()
    
    # Add a row for total
    total_row = pd.DataFrame([["Total", "", total_marks, obtained_marks, ""]], 
                             columns=["Question No.", "Question", "Max Marks", "Marks Obtained", "Justification"])
    df = pd.concat([df, total_row], ignore_index=True)
    
    # Save the processed summary for debugging
    try:
        debug_summary_path = eval_output_dir / "processed_summary.csv"
        df.to_csv(debug_summary_path, index=False)
        logger.debug(f"Saved processed summary to {debug_summary_path}")
    except Exception as e:
        logger.warning(f"Could not save processed summary for debugging: {e}")
    
    return df

def generate_aggregate_summary(all_summaries, output_dir):
    """Generate an aggregate summary from multiple evaluations using a voting-based approach.
    
    Args:
        all_summaries: List of evaluation summary dataframes
        output_dir: Directory to save the aggregate summary
    
    Returns:
        Dataframe containing aggregate summary with voting-based scores and all individual run scores
    """
    logger.info("Generating aggregate summary across all evaluations using voting system...")
    
    if not all_summaries:
        logger.warning("No evaluation summaries available for aggregation")
        return pd.DataFrame()
    
    # Take the first summary as a template (excluding the total row)
    template_df = all_summaries[0][:-1]
    
    # Define columns for the aggregate dataframe
    columns = ["Question No.", "Question", "Max Marks"]
    
    # Add columns for each run's score
    num_runs = len(all_summaries)
    for i in range(num_runs):
        columns.append(f"Run {i+1} Score")
    
    # Add aggregate columns
    columns.extend(["Voting Score", "Average Score", "Justification Notes"])
    
    # Create the aggregate dataframe
    aggregate_df = pd.DataFrame(columns=columns)
    
    # Calculate scores for each question
    for _, row in template_df.iterrows():
        q_no = row["Question No."]
        question = row["Question"]
        max_marks = row["Max Marks"]
        
        # Get scores for this question across all evaluations
        scores = [df.loc[df["Question No."] == q_no, "Marks Obtained"].values[0] 
                 for df in all_summaries if q_no in df["Question No."].values]
        
        # Ensure we have scores for the expected number of runs
        while len(scores) < num_runs:
            scores.append(0.0)
            logger.warning(f"Missing score for question {q_no} in at least one evaluation run")
        
        # Calculate voting-based score (mode)
        # Round scores to nearest 0.5 to group similar scores together for more effective voting
        rounded_scores = [round(score * 2) / 2 for score in scores]
        score_counts = {}
        for score in rounded_scores:
            score_counts[score] = score_counts.get(score, 0) + 1
        
        # Get the score(s) with the most votes
        max_vote_count = max(score_counts.values()) if score_counts else 0
        most_common_scores = [score for score, count in score_counts.items() if count == max_vote_count]
        
        # If there's a tie, use the highest score among the tied scores
        voting_score = max(most_common_scores) if most_common_scores else 0.0
        
        # Calculate average score as well
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Collect justifications from all evaluations
        justifications = [df.loc[df["Question No."] == q_no, "Justification"].values[0] 
                         for df in all_summaries if q_no in df["Question No."].values and "Justification" in df.columns]
        
        # Create a consolidated justification note with source run numbers
        justification_notes = []
        for i, justification in enumerate(justifications):
            if justification:
                justification_notes.append(f"Run {i+1}: {justification}")
        
        justification_note = " | ".join(justification_notes) if justification_notes else ""
        
        # Create row data with individual run scores
        row_data = {
            "Question No.": q_no,
            "Question": question,
            "Max Marks": max_marks,
            "Voting Score": round(voting_score, 1),
            "Average Score": round(avg_score, 2),
            "Justification Notes": justification_note
        }
        
        # Add individual run scores to the row
        for i, score in enumerate(scores):
            row_data[f"Run {i+1} Score"] = score
        
        # Add to aggregate DataFrame
        aggregate_df = pd.concat([aggregate_df, pd.DataFrame([row_data])], ignore_index=True)
    
    # Calculate totals
    total_max = aggregate_df["Max Marks"].sum()
    
    # Calculate total voting and average scores
    total_voting = aggregate_df["Voting Score"].sum()
    total_avg = aggregate_df["Average Score"].sum()
    
    # Calculate total for each run
    run_totals = {}
    for i in range(num_runs):
        run_column = f"Run {i+1} Score"
        run_totals[run_column] = aggregate_df[run_column].sum()
    
    # Create total row data
    total_data = {
        "Question No.": "Total",
        "Question": "",
        "Max Marks": total_max,
        "Voting Score": round(total_voting, 1),
        "Average Score": round(total_avg, 2),
        "Justification Notes": ""
    }
    
    # Add run totals
    for col, total in run_totals.items():
        total_data[col] = round(total, 1)
    
    # Add total row
    aggregate_df = pd.concat([aggregate_df, pd.DataFrame([total_data])], ignore_index=True)
    
    # Calculate percentage scores for better comparisons
    for i in range(num_runs):
        run_column = f"Run {i+1} Score"
        percentage_column = f"Run {i+1} %"
        aggregate_df[percentage_column] = aggregate_df.apply(
            lambda row: round(row[run_column] / row["Max Marks"] * 100, 1) if row["Max Marks"] > 0 else 0,
            axis=1
        )
    
    # Calculate percentage for voting and average scores
    aggregate_df["Voting %"] = aggregate_df.apply(
        lambda row: round(row["Voting Score"] / row["Max Marks"] * 100, 1) if row["Max Marks"] > 0 else 0,
        axis=1
    )
    
    aggregate_df["Average %"] = aggregate_df.apply(
        lambda row: round(row["Average Score"] / row["Max Marks"] * 100, 1) if row["Max Marks"] > 0 else 0,
        axis=1
    )
    
    # Save aggregate summary
    aggregate_csv_path = output_dir / "aggregate_evaluation_summary.csv"
    aggregate_df.to_csv(aggregate_csv_path, index=False)
    logger.info(f"Aggregate summary saved to {aggregate_csv_path}")
    
    # Also save a more readable HTML version with highlighting
    try:
        # Highlight highest and lowest scores
        def highlight_max_min(s):
            if s.name in [f"Run {i+1} Score" for i in range(num_runs)]:
                is_max = s == s.max()
                is_min = s == s.min()
                return ['background-color: #C6EFCE' if v else 
                        'background-color: #FFC7CE' if m else '' 
                        for v, m in zip(is_max, is_min)]
            return ['' for _ in range(len(s))]
        
        # Generate styled HTML
        html_path = output_dir / "aggregate_evaluation_summary.html"
        styled_df = aggregate_df.style.apply(highlight_max_min)
        
        # Save HTML file
        with open(html_path, 'w') as f:
            f.write("<html><head><title>Aggregate Evaluation Summary</title>")
            f.write("<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}</style>")
            f.write("</head><body>")
            f.write("<h2>Aggregate Evaluation Summary with Voting System</h2>")
            f.write(styled_df.to_html())
            f.write("</body></html>")
        
        logger.info(f"HTML aggregate summary saved to {html_path}")
    except Exception as e:
        logger.warning(f"Could not generate HTML summary: {e}")
    
    return aggregate_df 