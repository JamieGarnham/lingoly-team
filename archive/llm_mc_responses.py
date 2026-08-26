#!/usr/bin/env python3
"""
LLM Multiple Choice Judge for Evaluating Model Responses
Uses LLM as a judge to select the best answer from multiple choice options.
Based on llm_judge_responses.py but modified for multiple choice selection.
"""

import json
import os
import re
import time
import traceback
import csv
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter

import fire
from tqdm import tqdm

# Try to import prompt_models from the testing directory
sys.path.append('./testing/code')
try:
    import prompt_models
except ImportError:
    print("Warning: Could not import prompt_models. Make sure you're running from the correct directory.")
    sys.exit(1)

# Set CSV field size limit to system maximum
csv.field_size_limit(sys.maxsize)

# Constants
MAX_API_ATTEMPT = 10
OUTFOLDER = "./data/mc_judge_outputs"
TMP_PATH = "./data/mc_judge_outputs/tmp"
MODEL_LIST = "./testing/data/model_list.json"

def load_cache(model_name, dataset_name=None, format_filter=None, tmp_path=TMP_PATH, suffix="_mc"):
    """Load cached multiple choice judge responses"""
    cached = []
    cached_dict = {}
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            # Build expected filename pattern
            expected_pattern = model_name
            if dataset_name:
                expected_pattern += "_" + dataset_name
            if format_filter:
                expected_pattern += "_" + format_filter.lower()
            expected_pattern += suffix
            
            # Check if file matches the pattern
            if expected_pattern in file and "tmp" in file:
                with open(Path(root, file), "r", encoding='utf-8') as f:
                    cached.extend(json.load(f))
    
    for entry in cached:
        # Create unique key from row identifiers
        row_id = f"{entry['overall_question_n']}_{entry['question_n']}_{entry['serial']}"
        if row_id in cached_dict:
            print(f"Warning: Duplicate cache entry for {row_id}")
        cached_dict[row_id] = entry
    return cached_dict

def generate_repeat_answers(model_answers: List[str]) -> List[str]:
    """
    Generate set of answers that appear at least twice in the model's responses.
    Excludes invalid JSON and N/A responses.
    """
    # Filter out invalid responses
    valid_answers = []
    for answer in model_answers:
        if (answer and answer != "N/A" and 
            not (isinstance(answer, str) and answer.startswith("IMPROPER PARSING:"))):
            valid_answers.append(answer)
    
    # Count occurrences
    answer_counts = Counter(valid_answers)
    
    # Get answers that appear at least twice
    repeat_answers = [answer for answer, count in answer_counts.items() if count >= 2]
    
    return repeat_answers

def get_answer_frequencies(model_answers: List[str]) -> Dict[str, int]:
    """
    Get frequency count for all valid answers.
    """
    # Filter out invalid responses
    valid_answers = []
    for answer in model_answers:
        if (answer and answer != "N/A" and 
            not (isinstance(answer, str) and answer.startswith("IMPROPER PARSING:"))):
            valid_answers.append(answer)
    
    return dict(Counter(valid_answers))

def get_top_frequent_answers(answer_frequencies: Dict[str, int], max_options: int) -> List[str]:
    """
    Get the top X most frequent answers.
    """
    # Sort by frequency (descending) then alphabetically for ties
    sorted_answers = sorted(answer_frequencies.items(), key=lambda x: (-x[1], x[0]))
    return [answer for answer, _ in sorted_answers[:max_options]]

def extract_subquestion_text(problem_sheet: str, question_n: str, serial: str) -> str:
    """Extract the specific subquestion text from the problem sheet"""
    
    # Clean the problem sheet first
    cleaned_sheet = clean_problem_sheet(problem_sheet)
    
    # Strategy: Look for the question section and find what the serial is asking for
    
    # Since we've cleaned the problem sheet, we need to check the original for "Now respond" section
    # to extract the actual questions being asked
    now_respond_pattern = r'Now respond to the following questions:(.*?)(?=Make sure to finish|$)'
    now_respond_match = re.search(now_respond_pattern, problem_sheet, re.DOTALL | re.IGNORECASE)
    
    question_content = None
    
    if now_respond_match:
        # Use the "Now respond" section as primary content
        question_content = now_respond_match.group(1)
    else:
        # Fallback: Find the specific question section (e.g., "Q 1.1", "Q 5.2", etc.)
        question_section_pattern = rf'{re.escape(question_n)}[^Q]*?(?=Q\s+\d|\Z)'
        question_match = re.search(question_section_pattern, cleaned_sheet, re.DOTALL | re.IGNORECASE)
        if question_match:
            question_content = question_match.group(0)
    
    if question_content:
        
        # Look for the serial in this specific question section
        lines = question_content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Case 1: Line contains the serial followed by content (e.g., "a milkman", "b blind") 
            if line_stripped.startswith(serial + ' '):
                content = line_stripped[len(serial):].strip()
                
                # Check if this is a translation/task item (meaningful content)
                if content and len(content) > 3 and not re.match(r'^[A-Z]\s*$', content):
                    return f"{serial} {content}"
                
                # For single letter/short answers like "A", "B", "C" - look for question context
                elif content and len(content) <= 3:
                    # Look backwards for the question that this answers
                    for j in range(max(0, i-15), i):
                        prev_line = lines[j].strip()
                        if ('?' in prev_line or 
                            'give the answer' in prev_line.lower() or
                            'how would' in prev_line.lower() or
                            'how is it pronounced' in prev_line.lower()):
                            # Extract the main question
                            question_text = prev_line
                            return f"{serial} {question_text}"
            
            # Case 2: Line contains just the serial (e.g., "a" by itself)
            elif line_stripped == serial:
                # Look backwards for question context - prioritize more specific questions
                context_found = None
                for j in range(max(0, i-15), i):
                    prev_line = lines[j].strip()
                    
                    # Look for specific question types first
                    if 'which two words are they' in prev_line.lower():
                        context_found = prev_line
                    elif ('?' in prev_line and 
                          any(word in prev_line.lower() for word in ['how', 'what', 'give', 'pronounced'])):
                        if not context_found:  # Only use as fallback
                            context_found = prev_line
                
                if context_found:
                    return f"{serial} {context_found}"
    
    # Fallback: Look for common question patterns that include this serial
    
    # Pattern 1: "Which two words are they?" questions
    if re.search(r'Which two words are they\?', cleaned_sheet, re.IGNORECASE):
        which_two_pos = cleaned_sheet.lower().find('which two words are they?')
        if which_two_pos > -1:
            after_question = cleaned_sheet[which_two_pos:]
            if re.search(rf'\b{re.escape(serial)}\b', after_question[:200]):
                return f"{serial} Which two words are they?"
    
    # Pattern 2: Translation questions
    translate_patterns = [
        rf'Translate into Language X:[^{serial}]*{re.escape(serial)}\s+([^\n]+)',
        rf'Translate.*Language X.*{re.escape(serial)}\s+([^\n]+)',
    ]
    
    for pattern in translate_patterns:
        match = re.search(pattern, cleaned_sheet, re.IGNORECASE | re.DOTALL)
        if match:
            content = match.group(1).strip()
            if content:
                return f"{serial} {content}"
    
    # Pattern 3: Look for questions with dialectal variations
    dialect_patterns = [
        rf'how would this complex word be pronounced.*{re.escape(serial)}\s+([A-Z])',
        rf'how is it pronounced.*{re.escape(serial)}\s+([A-Z])',
        rf'how would you say.*{re.escape(serial)}\s+([A-Z])',
    ]
    
    for pattern in dialect_patterns:
        match = re.search(pattern, cleaned_sheet, re.IGNORECASE | re.DOTALL)
        if match:
            dialect = match.group(1)
            return f"{serial} Dialect {dialect}"
    
    # Pattern 4: General question extraction for single-character serials  
    lines = cleaned_sheet.split('\n')
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Look for lines with just the serial
        if line_stripped == serial:
            # Search backwards for a question
            found_question = None
            for j in range(max(0, i-20), i):
                prev_line = lines[j].strip()
                if ('?' in prev_line and 
                    any(word in prev_line.lower() for word in 
                        ['how', 'what', 'which', 'give', 'pronounced', 'differ'])):
                    found_question = prev_line
                    break
            
            if found_question:
                return f"{serial} {found_question}"
    
    # Fallback if no specific content found
    return f"Answer question {serial}."

def clean_problem_sheet(problem_sheet: str) -> str:
    """Clean the problem sheet by removing duplicate instructions and JSON format requirements"""
    
    # Remove the duplicate opening instruction
    duplicate_instruction = "Below is a problem sheet from a lingusitics exam. You will first see the entire sheet, then be asked to respond to specific questions from the sheet. Your answers to the questions should rely only on reasoning about the information provided in the sheet."
    
    if problem_sheet.startswith(duplicate_instruction):
        problem_sheet = problem_sheet[len(duplicate_instruction):].strip()
    
    # Remove the "Now respond to the following questions:" section and everything after it
    now_respond_pattern = r'Now respond to the following questions:.*$'
    problem_sheet = re.sub(now_respond_pattern, '', problem_sheet, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Remove JSON format instructions at the end (in case they appear before the "Now respond" section)
    json_instruction_patterns = [
        r'Make sure to finish your answer with json output with the keys as provided below:\s*\{[^}]*\}\s*',
        r'Make sure to finish your answer with json output[^{]*\{[^}]*\}\s*',
        r'\{[^}]*"a\."[^}]*\}\s*$'
    ]
    
    for pattern in json_instruction_patterns:
        problem_sheet = re.sub(pattern, '', problem_sheet, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    return problem_sheet.strip()

def create_mc_judge_prompt(full_problem_sheet: str, question_n: str, serial: str, options: List[str]) -> str:
    """Create the multiple choice judge prompt"""
    
    # Clean the problem sheet
    cleaned_problem_sheet = clean_problem_sheet(full_problem_sheet)
    
    # Extract the specific subquestion text
    subquestion_text = extract_subquestion_text(full_problem_sheet, question_n, serial)
    
    # Parse the subquestion to separate question text from serial content
    # Format: "serial question_text" or "serial content" 
    parts = subquestion_text.split(' ', 1)
    extracted_serial = parts[0]
    
    if len(parts) > 1:
        content = parts[1]
        
        # Check if this is a full question (contains '?') or just task content
        if '?' in content:
            # This is a full question - use it as the question text
            question_text = content
            serial_content = ""
        else:
            # This is task content - need to find the actual question context
            # Look backwards from where we found this serial to find the question header
            question_text = f"Answer question {extracted_serial}:"  # Default fallback
            serial_content = content
            
            # Try to extract the actual question from the original problem sheet's "Now respond" section
            now_respond_pattern = r'Now respond to the following questions:(.*?)(?=Make sure to finish|$)'
            now_respond_match = re.search(now_respond_pattern, full_problem_sheet, re.DOTALL | re.IGNORECASE)
            
            if now_respond_match:
                now_respond_content = now_respond_match.group(1)
                lines = now_respond_content.split('\n')
                serial_line_idx = None
                
                # Find the line with our serial
                for i, line in enumerate(lines):
                    if line.strip().startswith(extracted_serial + ' ') or line.strip() == extracted_serial:
                        serial_line_idx = i
                        break
                
                if serial_line_idx is not None:
                    # Look backwards for question headers
                    for j in range(serial_line_idx - 1, max(0, serial_line_idx - 20), -1):
                        prev_line = lines[j].strip()
                        
                        # Look for question patterns
                        if ('translate' in prev_line.lower() and 
                            ':' in prev_line and 
                            len(prev_line) > 10):
                            question_text = prev_line.rstrip(':')
                            break
                        elif (prev_line.endswith(':') and 
                              any(word in prev_line.lower() for word in 
                                  ['translate', 'what', 'how', 'which', 'give', 'complete'])):
                            question_text = prev_line.rstrip(':')
                            break
    else:
        # Only serial, no content
        question_text = f"Answer question {extracted_serial}:"
        serial_content = ""
    
    # Sort options alphabetically
    sorted_options = sorted(options)
    options_text = "\n".join(sorted_options)
    
    # Build the prompt in the requested format
    prompt_parts = []
    
    # Preamble
    prompt_parts.append("Below is a problem sheet from a linguistics exam. You will first see the entire sheet, then be asked to respond to a specific subquestion from the sheet. You will be given a set of options to choose from. Your answers to the questions should rely only on reasoning about the information provided in the sheet.")
    
    # Entire puzzle sheet
    prompt_parts.append(cleaned_problem_sheet)
    
    # "Now provide the answer to the following subquestion."
    prompt_parts.append("Now provide the answer to the following subquestion.")
    
    # Question text
    prompt_parts.append(question_text)
    
    # Serial and serial text
    if serial_content:
        prompt_parts.append(f"{extracted_serial} {serial_content}")
    else:
        prompt_parts.append(extracted_serial)
    
    # "These are the options you have to choose from:"
    prompt_parts.append("These are the options you have to choose from:")
    
    # Options
    prompt_parts.append(options_text)
    
    # Post-amble
    # prompt_parts.append("Consider the logic that could be used to lead to each of the options presented, and show your reasoning for each option. Based on your reasoning, please select one option only that you think is the correct answer. If you think multiple options could be correct, select only one of them. Show your reasoning steps, but your output MUST end with a valid JSON dictionary in this exact format: {\"answer\": \"option\"} (where 'option' is the EXACT text of the option you have selected).")
    prompt_parts.append("Consider the logic that could be used to lead to each of the options presented. Based on your reasoning, please select one option only that you think is the correct answer. If you think multiple options could be correct, select only one of them. Your output MUST end with a valid JSON dictionary in this exact format: {\"answer\": \"option\"} (where 'option' is the EXACT text of the option you have selected).")

    # Join all parts with double newlines
    prompt = "\n\n".join(prompt_parts)
    
    return prompt

def parse_mc_judge_response(response: str, options: List[str]) -> Dict[str, Any]:
    """Parse the multiple choice judge's response and extract the selected answer"""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            if "answer" in result:
                selected_answer = result["answer"]
                
                # Verify the answer is in the options list
                if selected_answer in options:
                    return {
                        "selected_answer": selected_answer,
                        "valid": True
                    }
                else:
                    return {
                        "error": f"Selected answer '{selected_answer}' not in options list",
                        "selected_answer": selected_answer,
                        "valid": False
                    }
            else:
                return {"error": "No 'answer' key found in JSON response", "valid": False}
        else:
            return {"error": "No JSON found in response", "valid": False}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}", "valid": False}
    except Exception as e:
        return {"error": f"Parse error: {str(e)}", "valid": False}

def extract_dataset_name_from_path(csv_path: str) -> str:
    """Extract a clean dataset name from the CSV file path"""
    # Get the filename without extension
    filename = Path(csv_path).stem
    # Remove common prefixes/suffixes and clean up the name
    dataset_name = filename.replace('subquestion_eval_', '').replace('_eval', '').replace('evaluation_', '')
    # Replace underscores with dashes for consistency and limit length
    dataset_name = dataset_name.replace('_', '-')[:50]  # Limit to 50 chars
    return dataset_name

def load_evaluation_csv(csv_path: str, 
                       question_filter: int = None, 
                       serial_filter: str = None,
                       limit: int = None,
                       only_any_correct: bool = False,
                       min_unique_answers: int = 1,
                       repeat_answers_only: bool = False,
                       majority_threshold: float = None,
                       min_correct: int = None,
                       max_correct: int = None,
                       format_filter: str = None) -> List[Dict[str, Any]]:
    """Load the evaluation CSV and filter rows"""
    rows = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Set field size limit to system maximum for large text fields
        csv.field_size_limit(sys.maxsize)
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            # Apply filters
            if question_filter and int(row['overall_question_n']) != question_filter:
                continue
            if serial_filter and row['serial'] != serial_filter:
                continue
            
            # Filter by is_any_correct if requested
            if only_any_correct and row['is_any_correct'].lower() != 'true':
                continue
            
            # If repeat_answers_only is True, filter to rows where number_correct >= 2
            if repeat_answers_only and int(row['number_correct']) < 2:
                continue
            
            # Filter by majority_threshold if specified
            if majority_threshold is not None:
                total_responses = 32  # v1 through v32
                threshold_value = majority_threshold * total_responses
                if int(row['majority_size']) >= threshold_value:
                    continue
            
            # Filter by number of correct answers if requested
            if min_correct is not None or max_correct is not None:
                try:
                    number_correct = int(row['number_correct'])
                    if min_correct is not None and number_correct <= min_correct:
                        continue
                    if max_correct is not None and number_correct >= max_correct:
                        continue
                except (ValueError, KeyError):
                    print(f"Warning: Could not parse number_correct for row {i}")
                    continue
            
            # Filter by format_fixed if requested
            if format_filter is not None:
                try:
                    if row['format_fixed'] != format_filter:
                        continue
                except KeyError:
                    print(f"Warning: format_fixed column not found for row {i}")
                    continue
            
            # Get all model answers (v1 through v32)
            model_answers = []
            for j in range(1, 33):  # v1 through v32
                answer_key = f'model_answer_v{j}'
                if answer_key in row:
                    model_answers.append(row[answer_key])
            
            # Parse unique_answers from string representation
            try:
                unique_answers = ast.literal_eval(row['unique_answers'])
                if not isinstance(unique_answers, list):
                    print(f"Warning: unique_answers is not a list for row {i}")
                    continue
                
                # Filter out IMPROPER PARSING responses and N/A responses
                filtered_answers = []
                for answer in unique_answers:
                    if (answer != "N/A" and 
                        not (isinstance(answer, str) and answer.startswith("IMPROPER PARSING:"))):
                        filtered_answers.append(answer)
                
                if len(filtered_answers) < min_unique_answers:
                    continue
                    
                # Update the row with filtered answers
                unique_answers = filtered_answers
                
            except Exception as e:
                print(f"Warning: Could not parse unique_answers for row {i}: {e}")
                continue
            
            # Generate repeat answers
            repeat_answers = generate_repeat_answers(model_answers)
            
            # Get answer frequencies for top-X filtering
            answer_frequencies = get_answer_frequencies(model_answers)
            
            row['unique_answers_parsed'] = unique_answers
            row['repeat_answers'] = repeat_answers
            row['answer_frequencies'] = answer_frequencies
            row['model_answers'] = model_answers
            rows.append(row)
            
            if limit and len(rows) >= limit:
                break
    
    print(f"Loaded {len(rows)} rows for evaluation")
    return rows

def mc_judge_pipeline(
    model: str,
    response_data_csv: str = "openrouter_analysis/original_prompt/subquestion_eval_original_prompt_32.csv",
    model_list_path: str = MODEL_LIST,
    outfolder: str = OUTFOLDER,
    use_cache: bool = True,
    question_filter: int = None,
    serial_filter: str = None,
    limit: int = None,
    batch_size: int = 1,
    only_any_correct: bool = False,
    min_unique_answers: int = 1,
    repeat_answers_only: bool = False,
    max_options: int = None,
    majority_threshold: float = None,
    min_correct: int = None,
    max_correct: int = None,
    format_filter: str = None,
):
    """Main pipeline for multiple choice judging responses"""
    
    print(f"Starting LLM Multiple Choice Judge evaluation with model: {model}")
    
    # Load model details
    with open(model_list_path) as f:
        model_list = json.load(f)
    
    if model not in model_list:
        raise ValueError(f"Model {model} not found in model list")
    
    model_details = model_list[model]
    model_name = model_details["name"]
    
    # Load evaluation data
    print(f"Loading evaluation data from {response_data_csv}")
    rows = load_evaluation_csv(
        response_data_csv, 
        question_filter=question_filter,
        serial_filter=serial_filter,
        limit=limit,
        only_any_correct=only_any_correct,
        min_unique_answers=min_unique_answers,
        repeat_answers_only=repeat_answers_only,
        majority_threshold=majority_threshold,
        min_correct=min_correct,
        max_correct=max_correct,
        format_filter=format_filter
    )
    
    if len(rows) == 0:
        print("No rows to process. Exiting.")
        return
    
    # Setup output file with dataset name and format
    dataset_name = extract_dataset_name_from_path(response_data_csv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename with optional format
    filename_parts = [model_name.split('/')[-1], dataset_name]
    if format_filter:
        filename_parts.append(format_filter.lower())
    filename_parts.extend(["mc_judge_evaluation", timestamp])
    output_filename = "_".join(filename_parts) + ".jsonl"
    
    # Setup caching
    cached_dict = {}
    if use_cache:
        print("Loading cache...")
        cached_dict = load_cache(model_name.split('/')[-1], dataset_name=dataset_name, format_filter=format_filter)
        print(f"Found {len(cached_dict)} cached responses")
    
    # Create output directories
    Path(outfolder).mkdir(parents=True, exist_ok=True)
    Path(TMP_PATH).mkdir(parents=True, exist_ok=True)
    
    # Process rows
    results = []
    errors = []
    
    for i, row in enumerate(tqdm(rows, desc="MC judging responses")):
        row_id = f"{row['overall_question_n']}_{row['question_n']}_{row['serial']}"
        
        # Check cache first
        if use_cache and row_id in cached_dict:
            print(f"Using cached result for {row_id}")
            results.append(cached_dict[row_id])
            continue
        
        # Determine which answers to use as options
        if repeat_answers_only:
            options = row['repeat_answers']
        else:
            options = row['unique_answers_parsed']
        
        # Apply max_options filtering if specified
        if max_options and len(options) > max_options:
            # Get top frequent answers
            options = get_top_frequent_answers(row['answer_frequencies'], max_options)
        
        # Skip if no options available
        if len(options) == 0:
            print(f"No options available for {row_id}, skipping")
            continue
        
        # Get full problem sheet (questions field contains the entire problem)
        full_problem_sheet = row['questions']
        
        # Create multiple choice judge prompt
        mc_judge_prompt = create_mc_judge_prompt(full_problem_sheet, row['question_n'], row['serial'], options)
        
        # Prepare batch for model calling
        batch = {
            "questions": [mc_judge_prompt],
            "answers": [{}],  # Not needed for judging
            "index": [[row['overall_question_n'], row['question_n'], row['serial']]],
            "metadata": [{"row_id": row_id}],
        }
        
        # Call model with retry logic
        attempt = 0
        success = False
        
        while attempt < MAX_API_ATTEMPT and not success:
            try:
                if model_details["model_type"] in ["openai", "anthropic", "cohere", "google", "open_router"]:
                    responses, raw_output = prompt_models.prompt_closed_model(
                        batch, model_details, cot=False
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_details['model_type']}")
                
                # Parse multiple choice judge response
                parse_result = parse_mc_judge_response(raw_output, options)
                
                # Extract subquestion for result storage
                subquestion_text = extract_subquestion_text(full_problem_sheet, row['question_n'], row['serial'])
                
                # Create result entry
                result = {
                    "overall_question_n": int(row['overall_question_n']),
                    "question_n": row['question_n'],
                    "serial": row['serial'],
                    "row_id": row_id,
                    "full_problem_sheet": full_problem_sheet,
                    "subquestion": subquestion_text,
                    "options": options,
                    "repeat_answers": row['repeat_answers'],
                    "unique_answers": row['unique_answers_parsed'],
                    "answer_frequencies": row['answer_frequencies'],
                    "correct_answer": row['correct_answer'],
                    "mc_judge_prompt": mc_judge_prompt,
                    "mc_judge_raw_response": raw_output,
                    "parse_result": parse_result,
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "repeat_answers_only": repeat_answers_only,
                    "max_options": max_options,
                }
                
                results.append(result)
                success = True
                
            except Exception as e:
                print(f"Error in API call for {row_id}, attempt {attempt + 1}: {str(e)}")
                traceback.print_exc()
                attempt += 1
                
                if attempt < MAX_API_ATTEMPT:
                    sleep_time = min(10 * attempt, 60)  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    error_entry = {
                        "row_id": row_id,
                        "error": str(e),
                        "options": options,
                    }
                    errors.append(error_entry)
                    print(f"Failed to process {row_id} after {MAX_API_ATTEMPT} attempts")
        
        # Save intermediate results every 5 rows
        if (i + 1) % 5 == 0:
            tmp_filename = f"{output_filename.replace('.jsonl', '')}_tmp_{i+1}.json"
            tmp_path = Path(TMP_PATH) / tmp_filename
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(results[-5:], f, indent=2, ensure_ascii=False)
                print(f"Saved temporary results to {tmp_path}")
            except Exception as e:
                print(f"Warning: Could not save temporary results: {e}")
    
    # Save final results
    output_path = Path(outfolder) / output_filename
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Saved {len(results)} results to {output_path}")
    except Exception as e:
        print(f"Error saving final results: {e}")
        # Try saving one by one
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                try:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                except Exception as e2:
                    print(f"Could not save result {result.get('row_id', 'unknown')}: {e2}")
    
    # Save errors if any
    if errors:
        error_path = Path(outfolder) / f"{output_filename.replace('.jsonl', '')}_errors.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(errors)} errors to {error_path}")
    
    print(f"MC Judge evaluation complete! Processed {len(results)} rows with {len(errors)} errors.")

if __name__ == "__main__":
    fire.Fire(mc_judge_pipeline)