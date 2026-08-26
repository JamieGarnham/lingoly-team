#!/usr/bin/env python3
"""
Script to process DeepSeek R1 shuffle benchmark results:
1. Rename files with version prefixes (v1_, v2_, etc.)
2. Move them to openrouter_runs/deepseek_shuffle/ 
3. Evaluate responses at subquestion level with individual correctness
"""

import json
import csv
import os
import shutil
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

def collect_and_move_files():
    """
    Collect R1 shuffle response files from testing folders and move them to openrouter_runs/deepseek_shuffle
    """
    base_dir = Path("/Users/jamiegarnham/lingoly2")
    target_dir = base_dir / "openrouter_runs" / "deepseek_shuffle"
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    moved_files = []
    
    # Process testing_3 through testing_18 (v1 through v16)
    for i in range(3, 19):
        version_num = i - 2  # testing_3 = v1, testing_4 = v2, etc.
        testing_dir = base_dir / f"testing_{i}"
        source_file = testing_dir / "data" / "responses_obf" / "deepseek-r1_lingoly_shuffle_filtered.json"
        
        if source_file.exists():
            target_filename = f"v{version_num}_deepseek-r1_lingoly_shuffle_filtered.json"
            target_file = target_dir / target_filename
            
            # Copy file to target location
            shutil.copy2(source_file, target_file)
            moved_files.append((version_num, target_file))
            print(f"Copied testing_{i} -> {target_filename}")
        else:
            print(f"Warning: {source_file} not found")
    
    print(f"\nMoved {len(moved_files)} files to {target_dir}")
    return moved_files, target_dir

def load_responses_from_directory(target_dir):
    """
    Load all response files from the target directory
    """
    responses_by_version = {}
    
    # Find all v*_*.json files
    for file_path in target_dir.glob("v*_*.json"):
        match = re.match(r'v(\d+)_.*\.json', file_path.name)
        if match:
            version_num = int(match.group(1))
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
    Main function to process DeepSeek R1 shuffle results
    """
    parser = argparse.ArgumentParser(description='Process DeepSeek R1 shuffle benchmark results')
    parser.add_argument('--skip-move', action='store_true', 
                       help='Skip file moving step (files already moved)')
    
    args = parser.parse_args()
    
    # Step 1: Collect and move files (unless skipped)
    if not args.skip_move:
        print("Step 1: Collecting and moving R1 shuffle response files...")
        moved_files, target_dir = collect_and_move_files()
    else:
        target_dir = Path("/Users/jamiegarnham/lingoly2/openrouter_runs/deepseek_shuffle")
        print(f"Step 1: Skipped. Using existing files in {target_dir}")
    
    # Step 2: Load exam papers format mapping
    print("\nStep 2: Loading exam papers format mapping...")
    format_mapping = load_exam_papers_format()
    
    # Step 3: Load responses from all versions
    print("\nStep 3: Loading responses from all versions...")
    responses_by_version = load_responses_from_directory(target_dir)
    print(f"Loaded responses from {len(responses_by_version)} versions: {list(responses_by_version.keys())}")
    
    # Step 4: Extract subquestion data with correctness
    print("\nStep 4: Extracting subquestion data with individual correctness...")
    rows = extract_subquestion_data_with_correctness(responses_by_version, format_mapping)
    
    # Step 5: Save to CSV
    print("\nStep 5: Saving to CSV...")
    
    # Create openrouter_analysis directory if it doesn't exist
    analysis_dir = Path("openrouter_analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate systematic filename
    num_versions = len(responses_by_version)
    output_filename = f"subquestion_eval_deepseek_shuffle_with_correctness_{num_versions}.csv"
    output_path = analysis_dir / output_filename
    
    save_to_csv(rows, output_path)
    
    print(f"\nDone! Total rows: {len(rows)} (plus header)")
    print(f"Output saved to: {output_path}")
    print(f"Processed {num_versions} versions of DeepSeek R1 shuffle responses")

if __name__ == "__main__":
    main()