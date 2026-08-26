#!/usr/bin/env python3
"""
LLM Judge for Evaluating Model Responses
Uses LLM as a judge to score unique answers from model responses.
Similar structure to benchmark_model.py for consistency.
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

import fire
from tqdm import tqdm

# Try to import prompt_models from the testing directory
import sys
sys.path.append('./testing/code')
try:
    import prompt_models
except ImportError:
    print("Warning: Could not import prompt_models. Make sure you're running from the correct directory.")
    sys.exit(1)

# Constants
MAX_API_ATTEMPT = 10
OUTFOLDER = "./data/judge_responses"
TMP_PATH = "./data/judge_responses/tmp"
MODEL_LIST = "./testing/data/model_list.json"

def load_cache(model_name, dataset_name=None, format_filter=None, tmp_path=TMP_PATH, suffix="_judge"):
    """Load cached judge responses"""
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

def parse_subquestion_from_text(questions_text: str, serial: str) -> str:
    """
    Extract the specific subquestion from the full question text.
    Removes preamble and focuses on the relevant subquestion.
    """
    # Step 1: Remove the preamble
    preamble_pattern = r"Below is a problem sheet.*?Your answers.*?sheet\."
    text_without_preamble = re.sub(preamble_pattern, "", questions_text, flags=re.DOTALL).strip()
    
    # Step 2: Extract the main question content (before "Now respond to the following questions:")
    main_question_match = re.search(r"(Question \d+:.*?)(?=\s*Now respond to the following questions:|$)", 
                                   text_without_preamble, re.DOTALL)
    
    if main_question_match:
        main_question = main_question_match.group(1).strip()
    else:
        # Fallback: use everything before "Now respond"
        main_question = re.split(r"Now respond to the following questions:", text_without_preamble)[0].strip()
    
    # Step 3: Find the specific subquestion
    # First, find the "Now respond" section 
    now_respond_match = re.search(r"Now respond to the following questions:\s*(.*?)(?=Make sure to finish|$)", 
                                 questions_text, re.DOTALL)
    
    if now_respond_match:
        respond_section = now_respond_match.group(1).strip()
        
        # Find the question text and extract only the part relevant to this serial
        # Split by lines and find the question that contains our serial
        lines = respond_section.split('\n')
        
        current_question = []
        found_serial = False
        in_target_question = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts with our serial
            if re.match(rf'^{re.escape(serial)}\s', line):
                found_serial = True
                subquestion_content = line
                break
                
            # Check if this line has a question followed by our serial
            if f"{serial} " in line or line.endswith(f"{serial}"):
                # Found the question that has our serial
                # Extract the question part and the serial part
                if line.endswith(f"{serial}"):
                    question_part = re.sub(rf'\s*{serial}\s*$', '', line).strip()
                    subquestion_content = f"{question_part}\n{serial}"
                else:
                    subquestion_content = line
                found_serial = True
                break
        
        if not found_serial:
            # Fallback: just note we're looking for this serial
            subquestion_content = f"Answer question {serial}"
    else:
        subquestion_content = f"Answer question {serial}"
    
    # Combine main question with the specific subquestion
    puzzle_text = f"{main_question}\n\n{subquestion_content}"
    
    return puzzle_text.strip()

def create_judge_prompt(puzzle_text: str, unique_answers: List[str]) -> str:
    """Create the judge prompt for evaluating answers"""
    
    # Format the answers list
    answers_list = []
    for i, answer in enumerate(unique_answers, 1):
        answers_list.append(f"{i}. {answer}")
    
    answers_section = "\n".join(answers_list)
    
    # Create the JSON format template
    json_template = "{\n"
    for i in range(1, len(unique_answers) + 1):
        json_template += f'  "answer_{i}": 0.0'
        if i < len(unique_answers):
            json_template += ","
        json_template += "\n"
    json_template += "}"
    
    prompt = f"""Evaluate the following solutions to a linguistic puzzle and return ONLY a JSON dictionary with scores.

PUZZLE: {puzzle_text}

POSSIBLE ANSWERS:
{answers_section}

Evaluate each answer's correctness and assign a ranking to each one, where 1 is the answer that you think is most likely to be correct, 2 is the next most likely, etc.

Consider:
- Whether each answer could satisfy the puzzle constraints
- The possible logical reasoning behind each answer
- How well it addresses what the puzzle is asking

Output ONLY a valid JSON dictionary in this exact format:
{json_template}

Use the exact answer numbers as keys. Include ALL answers. Output no other text."""
    
    return prompt

def parse_judge_response(response: str, num_answers: int) -> Dict[str, float]:
    """Parse the judge's response and extract scores"""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            scores = json.loads(json_str)
            
            # Validate that all expected keys are present
            expected_keys = {f"answer_{i}" for i in range(1, num_answers + 1)}
            actual_keys = set(scores.keys())
            
            if expected_keys == actual_keys:
                # Convert all values to float
                return {k: float(v) for k, v in scores.items()}
            else:
                print(f"Warning: Missing keys. Expected {expected_keys}, got {actual_keys}")
                return {"error": "Missing or extra keys in response"}
        else:
            return {"error": "No JSON found in response"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}"}
    except Exception as e:
        return {"error": f"Parse error: {str(e)}"}

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
            
            row['unique_answers_parsed'] = unique_answers
            rows.append(row)
            
            if limit and len(rows) >= limit:
                break
    
    print(f"Loaded {len(rows)} rows for evaluation")
    return rows

def judge_pipeline(
    model: str,
    response_data_csv: str = "subquestion_evaluation.csv",
    model_list_path: str = MODEL_LIST,
    outfolder: str = OUTFOLDER,
    use_cache: bool = True,
    question_filter: int = None,
    serial_filter: str = None,
    limit: int = None,
    batch_size: int = 1,
    only_any_correct: bool = False,
    min_unique_answers: int = 1,
    min_correct: int = None,
    max_correct: int = None,
    format_filter: str = None,
):
    """Main pipeline for judging responses"""
    
    print(f"Starting LLM Judge evaluation with model: {model}")
    
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
    filename_parts.extend(["judge_evaluation", timestamp])
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
    
    for i, row in enumerate(tqdm(rows, desc="Judging responses")):
        row_id = f"{row['overall_question_n']}_{row['question_n']}_{row['serial']}"
        
        # Check cache first
        if use_cache and row_id in cached_dict:
            print(f"Using cached result for {row_id}")
            results.append(cached_dict[row_id])
            continue
        
        # Parse subquestion
        puzzle_text = parse_subquestion_from_text(row['questions'], row['serial'])
        unique_answers = row['unique_answers_parsed']
        
        # Create judge prompt
        judge_prompt = create_judge_prompt(puzzle_text, unique_answers)
        
        # Prepare batch for model calling (similar to benchmark_model.py)
        batch = {
            "questions": [judge_prompt],
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
                
                # Parse judge response
                scores = parse_judge_response(raw_output, len(unique_answers))
                
                # Create result entry
                result = {
                    "overall_question_n": int(row['overall_question_n']),
                    "question_n": row['question_n'],
                    "serial": row['serial'],
                    "row_id": row_id,
                    "puzzle_text": puzzle_text,
                    "unique_answers": unique_answers,
                    "judge_prompt": judge_prompt,
                    "judge_raw_response": raw_output,
                    "scores": scores,
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
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
                        "unique_answers": unique_answers,
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
    
    print(f"Judge evaluation complete! Processed {len(results)} rows with {len(errors)} errors.")

if __name__ == "__main__":
    fire.Fire(judge_pipeline)