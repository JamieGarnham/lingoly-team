#!/usr/bin/env python3
"""
Script to evaluate chained prompt model responses at the subquestion level.
Creates a CSV where each row corresponds to a subquestion with all model answers.
"""

import json
import csv
import os
import sys
import argparse
from pathlib import Path
import ast
import re
import random
from collections import Counter

def normalize_answer(answer):
    """
    Normalize an answer for comparison by stripping whitespace and full stops,
    and converting to lowercase.
    """
    if not isinstance(answer, str):
        answer = str(answer)
    return answer.strip().rstrip('.').lower()

def is_valid_model_answer(answer):
    """
    Check if a model answer is valid (not N/A, Invalid JSON, or empty).
    """
    if not answer or answer == "N/A":
        return False
    
    # Convert to string if it's not already
    if not isinstance(answer, str):
        answer = str(answer)
    
    answer_lower = answer.lower().strip()
    if "invalid json" in answer_lower or answer_lower == "":
        return False
    return True

def calculate_majority_and_tiebreaker(model_answers):
    """
    Calculate majority answer and tiebreaker from a list of model answers.
    Returns: (majority, tiebreaker, majority_size)
    """
    # Set fixed random seed for reproducible results
    random.seed(42)
    
    # Filter out invalid answers and convert to strings
    valid_answers = []
    for ans in model_answers:
        if is_valid_model_answer(ans):
            # Convert to string if needed (for Counter to work with lists/objects)
            if isinstance(ans, (list, dict)):
                ans = str(ans)
            valid_answers.append(ans)
    
    if not valid_answers:
        return "N/A", "N/A", 0
    
    # Count occurrences
    answer_counts = Counter(valid_answers)
    max_count = max(answer_counts.values())
    
    # Find all answers with maximum count
    majority_answers = [ans for ans, count in answer_counts.items() if count == max_count]
    
    # Determine majority and tiebreaker
    if len(majority_answers) == 1:
        majority = majority_answers[0]
        tiebreaker = majority
    else:
        majority = "N/A"  # Tie, so no clear majority
        tiebreaker = random.choice(majority_answers)  # Random selection from tied answers
    
    majority_size = max_count
    
    return majority, tiebreaker, majority_size

def check_answer_correctness(answer, correct_answers):
    """
    Check if an answer matches any of the correct answers.
    """
    if not is_valid_model_answer(answer):
        return False
    
    normalized_answer = normalize_answer(answer)
    normalized_correct = [normalize_answer(ca) for ca in correct_answers]
    
    return normalized_answer in normalized_correct

def parse_correct_answer(answer_value):
    """
    Parse correct answer which could be a string, list, or JSON string.
    """
    if isinstance(answer_value, list):
        return answer_value
    elif isinstance(answer_value, str):
        # Try to parse as JSON list first
        if answer_value.startswith('[') and answer_value.endswith(']'):
            try:
                return json.loads(answer_value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(answer_value)
                except:
                    return [answer_value]
        else:
            return [answer_value]
    else:
        return [str(answer_value)]

def load_exam_papers_format():
    """
    Load the past-exam-papers.csv to get question formats.
    """
    exam_papers_path = Path("testing/data/past-exam-papers.csv")
    if not exam_papers_path.exists():
        print(f"Warning: {exam_papers_path} not found. Using empty format mapping.")
        return {}
    
    format_mapping = {}
    with open(exam_papers_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            overall_q_num = int(row['Overall Question Number'])
            question_format = row['Question Format']
            format_mapping[overall_q_num] = question_format
    
    return format_mapping

def load_chained_prompt_responses(chained_prompts_dir, sample_size=None):
    """
    Load chained prompt response files from the specified directory.
    
    Args:
        chained_prompts_dir: Path to the chained_prompts directory
        sample_size: If specified, randomly sample this many files
    """
    chained_dir = Path(chained_prompts_dir)
    if not chained_dir.exists():
        raise FileNotFoundError(f"Directory {chained_dir} does not exist")
    
    model_files = []
    
    # Find all files that match v*_*evaluation*.jsonl pattern
    for file_path in chained_dir.glob("v*_*evaluation*.jsonl"):
        match = re.match(r'v(\d+)_.*evaluation.*\.jsonl', file_path.name)
        if match:
            version_num = int(match.group(1))
            model_files.append((version_num, file_path))
    
    if not model_files:
        raise FileNotFoundError(f"No matching JSONL files found in {chained_dir}")
    
    # Sort by version number
    model_files.sort(key=lambda x: x[0])
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < len(model_files):
        # Set fixed random seed for reproducible results
        random.seed(42)
        model_files = random.sample(model_files, sample_size)
        # Re-sort after sampling
        model_files.sort(key=lambda x: x[0])
    
    responses_by_version = {}
    for version_num, file_path in model_files:
        version_key = f"v{version_num}"
        responses_by_version[version_key] = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    responses_by_version[version_key].append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line in {file_path}")
                    continue
    
    return responses_by_version

def extract_subquestion_data_from_chained(responses_by_version, format_mapping):
    """
    Extract subquestion data from all chained prompt model responses.
    """
    rows = []
    
    # First, collect all questions that appear in ALL versions to avoid N/A issues
    questions_by_version = {}
    for version_key, responses in responses_by_version.items():
        questions_by_version[version_key] = set()
        for response in responses:
            overall_question_n = response['overall_question_n']
            question_n = response['question_n']
            questions_by_version[version_key].add((overall_question_n, question_n))
    
    # Find questions that exist in ALL versions (intersection)
    common_questions = set.intersection(*questions_by_version.values()) if questions_by_version else set()
    print(f"Found {len(common_questions)} questions common to all {len(responses_by_version)} versions")
    
    # For each common question, collect serials using expected answers as authoritative source
    question_serials = {}  # Maps (overall_question_n, question_n) -> set of serials
    
    for version_key, responses in responses_by_version.items():
        for response in responses:
            overall_question_n = response['overall_question_n']
            question_n = response['question_n']
            
            # Only process questions that exist in all versions
            if (overall_question_n, question_n) not in common_questions:
                continue
            
            question_key = (overall_question_n, question_n)
            if question_key not in question_serials:
                question_serials[question_key] = set()
            
            # Use expected answers as authoritative source to avoid contamination from model responses
            expected_answer = response.get('expected_answer', {})
            
            # Add serials only from expected answers for this specific question
            question_specific_serials = set(expected_answer.keys())
            question_serials[question_key].update(question_specific_serials)
    
    # Now create subquestion combinations using the correct serials for each question
    subquestion_combinations = set()
    for (overall_question_n, question_n), serials in question_serials.items():
        for serial in serials:
            subquestion_combinations.add((overall_question_n, question_n, serial))
    
    # Process each unique subquestion
    for overall_question_n, question_n, serial in sorted(subquestion_combinations):
        # Get format from exam papers CSV
        question_format = format_mapping.get(overall_question_n, 'Unknown')
        
        # Initialize row
        row = {
            'questions': '',  # Will be filled from first available response
            'overall_question_n': overall_question_n,
            'question_n': question_n,
            'serial': serial,
            'format': question_format,
            'correct_answer': []
        }
        
        # Collect model answers for this subquestion
        model_answers_list = []
        
        for version_key in sorted(responses_by_version.keys(), key=lambda x: int(x[1:])):
            version_responses = responses_by_version[version_key]
            
            # Find the corresponding question in this version
            version_answer = "N/A"
            for response in version_responses:
                if (response['overall_question_n'] == overall_question_n and 
                    response['question_n'] == question_n):
                    
                    # Fill in question text and correct answer if not already set
                    if not row['questions']:
                        # Extract question text from the response if available
                        question_details = response.get('question_details', {})
                        if 'context' in question_details.get('metadata', {}):
                            row['questions'] = question_details['metadata']['context'][:200] + "..."  # Truncate for CSV
                    
                    if not row['correct_answer']:
                        expected = response.get('expected_answer', {})
                        if serial in expected:
                            row['correct_answer'] = parse_correct_answer(expected[serial])
                    
                    # Get model answer
                    model_parsed = response.get('model_parsed_response', {})
                    if serial in model_parsed:
                        version_answer = model_parsed[serial]
                    break
            
            row[f'model_answer_{version_key}'] = version_answer
            model_answers_list.append(version_answer)
        
        # Calculate majority and tiebreaker
        majority, tiebreaker, majority_size = calculate_majority_and_tiebreaker(model_answers_list)
        row['majority'] = majority
        row['tiebreaker'] = tiebreaker
        row['majority_size'] = majority_size
        
        # Get unique answers (excluding N/A and invalid responses)
        unique_answers = []
        seen = set()
        for ans in model_answers_list:
            if is_valid_model_answer(ans):
                # Convert to string for comparison but preserve original format
                ans_str = str(ans) if not isinstance(ans, str) else ans
                if ans_str not in seen:
                    unique_answers.append(ans_str)
                    seen.add(ans_str)
        row['unique_answers'] = unique_answers
        
        # Check correctness
        correct_answers_list = row['correct_answer']
        row['is_majority_correct'] = check_answer_correctness(majority, correct_answers_list)
        row['is_tiebreaker_correct'] = check_answer_correctness(tiebreaker, correct_answers_list)
        
        # Check if any model answer is correct
        any_correct = any(check_answer_correctness(ans, correct_answers_list) for ans in model_answers_list)
        row['is_any_correct'] = any_correct
        
        # Count how many are correct
        number_correct = sum(1 for ans in model_answers_list if check_answer_correctness(ans, correct_answers_list))
        row['number_correct'] = number_correct
        
        rows.append(row)
    
    return rows

def save_to_csv(rows, output_path):
    """
    Save the extracted data to CSV file.
    """
    if not rows:
        print("No data to save.")
        return
    
    # Get all column names in the desired order
    columns = ['questions', 'overall_question_n', 'question_n', 'serial', 'format', 'correct_answer']
    
    # Add model answer columns (sorted by version number)
    model_columns = [col for col in rows[0].keys() if col.startswith('model_answer_')]
    model_columns.sort(key=lambda x: int(x.split('_v')[1]) if '_v' in x else int(x.split('_')[2][1:]))
    columns.extend(model_columns)
    
    # Add analysis columns
    analysis_columns = ['majority', 'tiebreaker', 'majority_size', 'unique_answers', 'is_majority_correct', 
                       'is_tiebreaker_correct', 'is_any_correct', 'number_correct']
    columns.extend(analysis_columns)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved {len(rows)} rows to {output_path}")

def main():
    """
    Main function to process all chained prompt responses and create evaluation CSV.
    """
    parser = argparse.ArgumentParser(description='Evaluate chained prompt responses at the subquestion level')
    parser.add_argument('chained_prompts_dir', help='Path to the chained_prompts directory')
    parser.add_argument('sample_size', type=int, help='Number of response files to randomly sample')
    
    args = parser.parse_args()
    
    print(f"Loading exam papers format mapping...")
    format_mapping = load_exam_papers_format()
    
    print(f"Loading chained prompt responses from {args.chained_prompts_dir}...")
    responses_by_version = load_chained_prompt_responses(args.chained_prompts_dir, args.sample_size)
    print(f"Loaded responses from {len(responses_by_version)} versions: {list(responses_by_version.keys())}")
    
    print("Extracting subquestion data...")
    rows = extract_subquestion_data_from_chained(responses_by_version, format_mapping)
    
    print("Saving to CSV...")
    # Create openrouter_analysis directory if it doesn't exist
    analysis_dir = Path("openrouter_analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate systematic filename
    output_filename = f"subquestion_eval_chained_prompt_rosetta_{args.sample_size}.csv"
    output_path = analysis_dir / output_filename
    
    save_to_csv(rows, output_path)
    
    print(f"Done! Total rows: {len(rows)} (plus header)")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()