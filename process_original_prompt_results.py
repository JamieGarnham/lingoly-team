#!/usr/bin/env python3
"""
Script to process first 16 original prompt DeepSeek R1 responses:
1. Load the first 16 versions from openrouter_runs/original_prompt/
2. Evaluate responses at subquestion level with individual correctness
3. Generate CSV similar to the shuffle analysis
"""

import json
import csv
import os
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

def load_original_prompt_responses(source_dir, num_versions=16):
    """
    Load the first N versions from openrouter_runs/original_prompt directory
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Directory {source_path} does not exist")
    
    responses_by_version = {}
    
    # Get all version files and sort them numerically
    version_files = []
    for file_path in source_path.glob("v*_deepseek-r1_lingoly.json"):
        match = re.match(r'v(\d+)_deepseek-r1_lingoly\.json', file_path.name)
        if match:
            version_num = int(match.group(1))
            version_files.append((version_num, file_path))
    
    # Sort by version number and take first num_versions
    version_files.sort(key=lambda x: x[0])
    version_files = version_files[:num_versions]
    
    if len(version_files) < num_versions:
        print(f"Warning: Only found {len(version_files)} files, requested {num_versions}")
    
    # Load the selected versions
    for version_num, file_path in version_files:
        version_key = f"v{version_num}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            responses_by_version[version_key] = json.load(f)
        
        print(f"Loaded {version_key}: {len(responses_by_version[version_key])} responses")
    
    return responses_by_version

def extract_subquestion_data_with_correctness(responses_by_version, format_mapping):
    """
    Extract subquestion data from all model responses with individual correctness columns.
    """
    rows = []
    
    # Use first version to get the question structure
    first_version = list(responses_by_version.keys())[0]
    questions = responses_by_version[first_version]
    
    for question_data in questions:
        overall_question_n = question_data['overall_question_n']
        question_n = question_data['question_n']
        questions_text = question_data['questions']
        
        # Get format from exam papers CSV
        question_format = format_mapping.get(overall_question_n, 'Unknown')
        
        # Get correct answers
        correct_answers = question_data['correct_answers'][0]  # First (and usually only) correct answer set
        
        # Process each subquestion (a, b, c, etc.)
        for serial, correct_answer in correct_answers.items():
            row = {
                'questions': questions_text,
                'overall_question_n': overall_question_n,
                'question_n': question_n,
                'serial': serial,
                'format': question_format,
                'correct_answer': parse_correct_answer(correct_answer)
            }
            
            # Add model answers and correctness for each version
            model_answers_list = []
            for version in sorted(responses_by_version.keys(), key=lambda x: int(x[1:])):
                version_responses = responses_by_version[version]
                
                # Find the corresponding question in this version
                version_question = None
                for q in version_responses:
                    if (q['overall_question_n'] == overall_question_n and 
                        q['question_n'] == question_n):
                        version_question = q
                        break
                
                if version_question and 'model_answers' in version_question:
                    model_answers = version_question['model_answers']
                    if isinstance(model_answers, dict) and serial in model_answers:
                        answer = model_answers[serial]
                        row[f'model_answer_{version}'] = answer
                        
                        # Check correctness for this specific version
                        correct_answers_list = parse_correct_answer(correct_answer)
                        is_correct = check_answer_correctness(answer, correct_answers_list)
                        row[f'is_{version}_correct'] = is_correct
                        
                        model_answers_list.append(answer)
                    else:
                        row[f'model_answer_{version}'] = "N/A"
                        row[f'is_{version}_correct'] = False
                        model_answers_list.append("N/A")
                else:
                    row[f'model_answer_{version}'] = "N/A"
                    row[f'is_{version}_correct'] = False
                    model_answers_list.append("N/A")
            
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
            correct_answers_list = parse_correct_answer(correct_answer)
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
    Save the extracted data to CSV file with interleaved model answers and correctness.
    """
    if not rows:
        print("No data to save.")
        return
    
    # Get all column names in the desired order
    columns = ['questions', 'overall_question_n', 'question_n', 'serial', 'format', 'correct_answer']
    
    # Add model answer and correctness columns (sorted by version number), interleaved
    model_answer_columns = [col for col in rows[0].keys() if col.startswith('model_answer_')]
    correctness_columns = [col for col in rows[0].keys() if col.startswith('is_v') and col.endswith('_correct')]
    
    # Extract version numbers and sort them
    versions = []
    for col in model_answer_columns:
        version_match = re.match(r'model_answer_v(\d+)', col)
        if version_match:
            versions.append(int(version_match.group(1)))
    
    versions = sorted(set(versions))
    
    # Interleave model_answer and is_correct columns
    for version in versions:
        columns.append(f'model_answer_v{version}')
        columns.append(f'is_v{version}_correct')
    
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
    Main function to process original prompt DeepSeek R1 responses
    """
    parser = argparse.ArgumentParser(description='Process original prompt DeepSeek R1 responses')
    parser.add_argument('--num-versions', type=int, default=16,
                       help='Number of versions to process (default: 16)')
    parser.add_argument('--source-dir', default='/Users/jamiegarnham/lingoly2/openrouter_runs/original_prompt',
                       help='Source directory containing response files')
    
    args = parser.parse_args()
    
    print(f"Processing first {args.num_versions} original prompt responses...")
    
    # Step 1: Load exam papers format mapping
    print("Step 1: Loading exam papers format mapping...")
    format_mapping = load_exam_papers_format()
    
    # Step 2: Load responses from specified number of versions
    print(f"Step 2: Loading responses from first {args.num_versions} versions...")
    responses_by_version = load_original_prompt_responses(args.source_dir, args.num_versions)
    print(f"Loaded responses from {len(responses_by_version)} versions: {list(responses_by_version.keys())}")
    
    # Step 3: Extract subquestion data with correctness
    print("Step 3: Extracting subquestion data with individual correctness...")
    rows = extract_subquestion_data_with_correctness(responses_by_version, format_mapping)
    
    # Step 4: Save to CSV
    print("Step 4: Saving to CSV...")
    
    # Create openrouter_analysis directory if it doesn't exist
    analysis_dir = Path("openrouter_analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate systematic filename
    num_versions = len(responses_by_version)
    output_filename = f"subquestion_eval_original_prompt_with_correctness_{num_versions}.csv"
    output_path = analysis_dir / output_filename
    
    save_to_csv(rows, output_path)
    
    print(f"\nDone! Total rows: {len(rows)} (plus header)")
    print(f"Output saved to: {output_path}")
    print(f"Processed {num_versions} versions of original prompt responses")

if __name__ == "__main__":
    main()