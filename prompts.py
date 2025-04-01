"""
Centralized storage for all prompts used in the automated answer script grading system.

This module contains the system and user prompts for all components:
- OCR: For extracting text from images
- Interpreter: For organizing OCR output by question
- Evaluation: For grading structured answers
"""

# Dictionary of machine learning domain-specific examples
ML_EXAMPLES = {
    "equations": [
        "[EQUATION] J(θ) = (1/2m) * Σ(h_θ(x^(i)) - y^(i))^2 where J is the cost function for linear regression [/EQUATION]",
        "[EQUATION] σ(z) = 1/(1+e^(-z)) where σ is the sigmoid activation function [/EQUATION]",
        "[EQUATION] P(y=1|x) = 1/(1 + e^(-(θ_0 + θ_1*x_1 + ... + θ_n*x_n))) for logistic regression [/EQUATION]",
    ],
    "figures": [
        "[FIGURE] A graph showing ROC curve with TPR on y-axis (0-1.0) and FPR on x-axis (0-1.0). The curve shows a concave shape above the diagonal line, with an AUC value of approximately 0.85 written next to it. [/FIGURE]",
        "[FIGURE] A confusion matrix diagram with four quadrants labeled: True Positive (45), False Positive (10), False Negative (5), and True Negative (40). Arrows indicate that precision = TP/(TP+FP) and recall = TP/(TP+FN). [/FIGURE]",
        "[FIGURE] A decision tree with the root node splitting on feature X > 0.7, with Gini impurity=0.4. The left branch leads to a leaf node for class A (samples=30), and the right branch splits further on feature Y > 0.5. [/FIGURE]",
    ],
    "calculations": [
        "[CALCULATION]\nprecision = TP/(TP+FP)\n= 45/(45+10)\n= 45/55\n= 0.818\n[/CALCULATION]",
        "[CALCULATION]\nF1 score = 2 * (precision * recall)/(precision + recall)\n= 2 * (0.818 * 0.9)/(0.818 + 0.9)\n= 2 * 0.736/1.718\n= 0.857\n[/CALCULATION]",
        "[CALCULATION]\nEntropy(S) = -Σ p_i * log_2(p_i)\n= -(0.6 * log_2(0.6) + 0.4 * log_2(0.4))\n= -(0.6 * (-0.737) + 0.4 * (-1.322))\n= 0.971\n[/CALCULATION]",
    ],
}

###################
# OCR PROMPTS
###################


def get_ocr_system_prompt(domain_context):
    """
    Generate the system prompt for OCR text extraction.

    Args:
        domain_context: The domain context of the exam (e.g., "machine learning exam")

    Returns:
        The system prompt for OCR text extraction
    """
    return f"""You are an OCR system specialized in extracting handwritten {domain_context} content.
Focus ONLY on accurate transcription of content and basic formatting markers.
Your job is to extract content exactly as written, not to organize or interpret it."""


def get_ocr_user_prompt(domain_context, include_examples=False):
    """
    Generate the user prompt for OCR text extraction.

    Args:
        domain_context: The domain context of the exam
        include_examples: Whether to include formatting examples

    Returns:
        The user prompt for OCR text extraction
    """
    # Create few-shot examples for complex elements if enabled
    examples = ""
    if include_examples:
        # Select one random example from each category
        import random

        selected_examples = [
            random.choice(ML_EXAMPLES["equations"]),
            random.choice(ML_EXAMPLES["figures"]),
            random.choice(ML_EXAMPLES["calculations"]),
        ]

        examples = f"""
EXAMPLES OF FORMATTING:

Mathematical equation:
{selected_examples[0]}

Figure description:
{selected_examples[1]}

Calculation steps:
{selected_examples[2]}
"""

    return f"""Transcribe all handwritten content from this exam answer image.

YOUR RESPONSIBILITIES:
1. Extract all visible text exactly as written
2. Mark special content with appropriate tags:
   - Equations: [EQUATION]...[/EQUATION]  
   - Diagrams: [FIGURE]...[/FIGURE]
   - Calculations: [CALCULATION]...[/CALCULATION]
   - Illegible text: [ILLEGIBLE]
   - Uncertain text: [?text?]

3. Note basic structural elements:
   - When you see a question number, mark it as [Q#] on best effort basis
   - If content appears to continue from a previous page: [CONTINUES_FROM_PREVIOUS]
   - If content appears to continue to the next page: [CONTINUES_TO_NEXT]

4. Preserve mathematical notation:
   - Superscripts: x^2
   - Subscripts: x_i
   - Greek letters: [alpha], [beta], etc.
   - Fractions: (numerator)/(denominator)

{examples}

DO NOT attempt to organize content by question or section.
DO NOT skip any visible text, even if it appears irrelevant.
DO focus on preserving mathematical notation as accurately as possible.

This is one page of a multi-page exam."""


###################
# INTERPRETER PROMPTS
###################


def get_interpreter_system_prompt():
    """
    Generate the system prompt for interpreting OCR output.

    Returns:
        The system prompt for interpreting OCR output
    """
    return """You are an expert interpreter of OCR text from handwritten exam answers.
Your responsibility is to ORGANIZE and STRUCTURE the raw OCR output into a coherent, question-by-question format.

Your key responsibilities:
1. Identify and separate each exam question
2. Connect content that spans across multiple pages
3. Organize special elements (equations, figures, calculations) within their proper questions
4. Normalize formatting variations across the document
5. Ensure all content is preserved and assigned to the correct question

You should NOT evaluate the correctness of answers or assign scores.
You should NOT attempt to improve or correct the content - preserve it exactly as provided.
You should FOCUS on creating a clean, organized structure that will be easy to evaluate."""


def get_interpreter_user_prompt(all_ocr_content):
    """
    Generate the user prompt for interpreting OCR output.

    Args:
        all_ocr_content: The OCR content to interpret

    Returns:
        The user prompt for interpreting OCR output
    """
    return f"""Organize the following OCR-extracted text from exam answer sheets into a structured format:

{all_ocr_content}

YOUR TASK:
1. Analyze all pages to identify distinct questions (marked with [Q#])
2. Group content by question number across all pages
3. Connect content marked with [CONTINUES_FROM_PREVIOUS] or [CONTINUES_TO_NEXT]
4. Extract and organize special elements:
   - Equations (marked with [EQUATION]...[/EQUATION])
   - Figures (marked with [FIGURE]...[/FIGURE])
   - Calculations (marked with [CALCULATION]...[/CALCULATION])
5. Ensure ALL content is preserved and properly assigned

Return a structured JSON with the following format:

```json
{{
  "questions": {{
    "1": {{
      "question_number": "1",
      "raw_text": "The full raw answer text...",
      "equations": ["equation 1", "equation 2", ...],
      "figures": ["figure description 1", ...],
      "calculations": ["calculation steps 1", ...],
      "continues_from_previous": false,
      "continues_to_next": true
    }},
    "2": {{
      // Question 2 data...
    }},
    // Additional questions...
  }},
  "metadata": {{
    "total_questions": 3,
    "pages_processed": 5
  }},
  "summary": "Brief overall summary of answer content"
}}
```

IMPORTANT GUIDELINES:
- Preserve the exact wording from the OCR output
- Do not evaluate or judge the quality of answers
- If content cannot be assigned to a specific question, place it in an "unassigned" question
- Pay special attention to connecting content that spans multiple pages"""


###################
# EVALUATION PROMPTS
###################


def get_evaluation_system_prompt(domain_context):
    """
    Generate the system prompt for evaluation.

    Args:
        domain_context: The domain context of the exam

    Returns:
        The system prompt for evaluation
    """
    return f"""You are an expert {domain_context} evaluator with years of experience grading technical exams.

YOUR RESPONSIBILITY:
Assess the quality and correctness of student answers based on subject matter expertise.
Focus ONLY on the content quality, not the formatting or organization.

You are the FINAL STEP in a processing pipeline:
1. OCR: Extracted text from handwritten pages
2. Interpreter: Organized content by question
3. YOU: Evaluate the correctness and quality of answers"""


def get_evaluation_user_prompt(
    domain_context, questions_table, answers, question_score_template
):
    """
    Generate the user prompt for evaluation.

    Args:
        domain_context: The domain context of the exam
        questions_table: The table of questions and maximum marks
        answers: The student answers to evaluate
        question_score_template: Template for scoring questions

    Returns:
        The user prompt for evaluation
    """
    return f"""Evaluate the following student answers for a {domain_context}.

Questions and Marks:
{questions_table}

Student Answers:
{answers}

EVALUATION CRITERIA:
- Correctness (70%): Factual accuracy and appropriate application of concepts
- Completeness (20%): Coverage of all required elements in the question
- Clarity (10%): Logical flow and coherence of ideas

SCORING GUIDELINES:
- Excellent (85-100%): Complete, correct answers demonstrating thorough understanding
- Satisfactory (65-85%): Mostly correct with minor omissions or errors
- Developing (35-65%): Partially correct with significant concepts missing
- Limited (0-35%): Major conceptual errors or incomplete responses

YOUR TASK:
For each question:
1. Assess the CONTENT QUALITY without being influenced by formatting artifacts
2. Identify correct technical concepts and misconceptions
3. Provide a justified score based on the evaluation criteria
4. Ignore any OCR or formatting artifacts when assessing the answer

IMPORTANT: After your detailed evaluation, provide a PERFECTLY FORMATTED score table with the EXACT format shown below:

<SCORE_TABLE>
Question,MaxMarks,MarksObtained,JustificationSummary
{question_score_template}
</SCORE_TABLE>

STRICT FORMATTING REQUIREMENTS:
1. Do NOT skip or omit any questions - ALL questions must be included in the table
2. Each row MUST have exactly 4 columns: Question number, MaxMarks, MarksObtained, and JustificationSummary
3. MarksObtained must be a number that doesn't exceed MaxMarks and is rounded to 1 decimal place when appropriate
4. Make sure every question's JustificationSummary is enclosed in double quotes
5. If a student didn't answer a question, score it as 0 and note "No answer provided" as justification
6. Ensure each line is complete and properly formatted - no partial rows or missing columns
7. Include a row for every question even if it received a score of 0

The table MUST be wrapped with <SCORE_TABLE> and </SCORE_TABLE> tags exactly as shown.

This table will be automatically parsed by a system, so precise formatting is critical."""
