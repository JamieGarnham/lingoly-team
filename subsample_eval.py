#!/usr/bin/env python3
"""
Script to create recursive subsamples of model responses and recalculate evaluation metrics.
Takes subsamples of 16 -> 8 -> 4 -> 2 -> 1 model responses and recalculates metrics for each.
"""

import csv
import sys
import random
import argparse
from pathlib import Path
from collections import Counter
import re

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

def check_answer_correctness(answer, correct_answers, overall_question_n=None, question_n=None, serial=None):
    """
    Check if an answer matches any of the correct answers.
    Includes special handling for specific questions.
    """
    if not is_valid_model_answer(answer):
        return False
    
    # Special case for question 5, Q 5.1, serial a
    if (overall_question_n == 5 and question_n == "Q 5.1" and serial == "a"):
        normalized_answer = normalize_answer(answer)
        # Check if answer contains both "üpgontüd" and "sopostüd"
        if "üpgontüd" in normalized_answer and "sopostüd" in normalized_answer:
            return True
    
    # Special case for question 170, Q5., k - add "langgbu'" and "maysu'" as correct
    if (overall_question_n == 170 and question_n == "Q 5." and serial == "k"):
        normalized_answer = normalize_answer(answer)
        # Check for langgbu or maysu with either apostrophe type (' or ')
        straight_apos = "'"  # ASCII 39
        curly_apos = chr(8217)  # Unicode 8217 '
        if (normalized_answer == f"langgbu{straight_apos}" or normalized_answer == f"langgbu{curly_apos}" or
            normalized_answer == f"maysu{straight_apos}" or normalized_answer == f"maysu{curly_apos}"):
            return True
    
    # Special case for question 75, Q7., 3 - add "two people who are not siblings"
    if (overall_question_n == 75 and question_n == "Q 7." and serial == "3"):
        normalized_answer = normalize_answer(answer)
        if "two people who are not siblings" in normalized_answer:
            return True
    
    # Standard correctness check
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
                import json
                return json.loads(answer_value)
            except json.JSONDecodeError:
                try:
                    import ast
                    return ast.literal_eval(answer_value)
                except:
                    return [answer_value]
        else:
            return [answer_value]
    else:
        return [str(answer_value)]

def subsample_columns(available_columns, target_size, seed=42):
    """
    Randomly subsample columns to target size.
    """
    random.seed(seed)
    if len(available_columns) <= target_size:
        return available_columns
    return random.sample(available_columns, target_size)

def recalculate_metrics(row, selected_versions, correct_answers_list, overall_question_n, question_n, serial):
    """
    Recalculate all metrics for the selected model versions.
    """
    # Get model answers for selected versions
    model_answers = []
    for version in selected_versions:
        answer = row.get(f'model_answer_{version}', 'N/A')
        model_answers.append(answer)
    
    # Calculate majority and tiebreaker
    majority, tiebreaker, majority_size = calculate_majority_and_tiebreaker(model_answers)
    
    # Get unique answers (excluding N/A and invalid responses)
    unique_answers = []
    seen = set()
    for ans in model_answers:
        if is_valid_model_answer(ans):
            # Convert to string for comparison but preserve original format
            ans_str = str(ans) if not isinstance(ans, str) else ans
            if ans_str not in seen:
                unique_answers.append(ans_str)
                seen.add(ans_str)
    
    # Check correctness
    is_majority_correct = check_answer_correctness(majority, correct_answers_list, overall_question_n, question_n, serial)
    is_tiebreaker_correct = check_answer_correctness(tiebreaker, correct_answers_list, overall_question_n, question_n, serial)
    
    # Check if any model answer is correct
    is_any_correct = any(check_answer_correctness(ans, correct_answers_list, overall_question_n, question_n, serial) for ans in model_answers)
    
    # Count how many are correct
    number_correct = sum(1 for ans in model_answers if check_answer_correctness(ans, correct_answers_list, overall_question_n, question_n, serial))
    
    return {
        'majority': majority,
        'tiebreaker': tiebreaker,
        'majority_size': majority_size,
        'unique_answers': unique_answers,
        'is_majority_correct': is_majority_correct,
        'is_tiebreaker_correct': is_tiebreaker_correct,
        'is_any_correct': is_any_correct,
        'number_correct': number_correct
    }

def process_subsample(input_file, output_prefix, subsample_sizes=[16, 8, 4, 2, 1]):
    """
    Process the input CSV to create subsamples and recalculate metrics.
    """
    # Increase CSV field size limit
    csv.field_size_limit(sys.maxsize)
    
    print(f"Loading {input_file}...")
    
    # Read the input file
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows)} rows")
    
    # Get all version numbers (v1-v32)
    all_versions = []
    for i in range(1, 33):
        if f'model_answer_v{i}' in rows[0]:
            all_versions.append(f'v{i}')
    
    print(f"Found {len(all_versions)} model versions: {all_versions[:5]}...{all_versions[-5:]}")
    
    # Create recursive subsamples
    current_versions = all_versions
    subsample_data = {}
    
    for size in subsample_sizes:
        print(f"Creating subsample of size {size}...")
        selected_versions = subsample_columns(current_versions, size)
        selected_versions.sort(key=lambda x: int(x[1:]))  # Sort by version number
        subsample_data[size] = selected_versions
        current_versions = selected_versions  # Use this subsample for the next iteration
        print(f"  Selected: {selected_versions}")
    
    # Process each subsample size separately
    for size in subsample_sizes:
        print(f"Processing subsample size {size}...")
        selected_versions = subsample_data[size]
        output_rows = []
        
        for row_idx, row in enumerate(rows):
            if row_idx % 200 == 0:
                print(f"  Processing row {row_idx + 1}/{len(rows)} for n{size}...")
            
            # Base information
            overall_question_n = int(row['overall_question_n'])
            question_n = row['question_n']
            serial = row['serial']
            correct_answers_list = parse_correct_answer(row['correct_answer'])
            
            output_row = {
                'questions': row['questions'],
                'overall_question_n': overall_question_n,
                'question_n': question_n,
                'serial': serial,
                'format': row['format'],
                'correct_answer': row['correct_answer']
            }
            
            # Add individual model answers and correctness for this subsample size
            for version in selected_versions:
                answer = row.get(f'model_answer_{version}', 'N/A')
                is_correct = check_answer_correctness(answer, correct_answers_list, overall_question_n, question_n, serial)
                
                output_row[f'model_answer_{version}'] = answer
                output_row[f'is_{version}_correct'] = is_correct
            
            # Calculate and add aggregate metrics
            metrics = recalculate_metrics(row, selected_versions, correct_answers_list, overall_question_n, question_n, serial)
            for key, value in metrics.items():
                output_row[key] = value
            
            output_rows.append(output_row)
        
        # Write output file for this subsample size
        output_file = f"{output_prefix}_{size}.csv"
        print(f"  Writing {len(output_rows)} rows to {output_file}...")
        
        # Create column order for this subsample
        columns = ['questions', 'overall_question_n', 'question_n', 'serial', 'format', 'correct_answer']
        
        # Add model answer and correctness columns
        for version in selected_versions:
            columns.append(f'model_answer_{version}')
            columns.append(f'is_{version}_correct')
        
        # Add aggregate metrics
        columns.extend([
            'majority', 'tiebreaker', 'majority_size', 'unique_answers', 
            'is_majority_correct', 'is_tiebreaker_correct', 'is_any_correct', 'number_correct'
        ])
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(output_rows)
        
        print(f"  Successfully created {output_file}")
        print(f"  Selected versions: {selected_versions}")
    
    print(f"\nCompleted all subsamples!")

def main():
    """
    Main function to process subsamples.
    """
    parser = argparse.ArgumentParser(description='Create recursive subsamples of model responses')
    parser.add_argument('input_file', help='Input CSV file (subquestion_eval_*_32.csv)')
    parser.add_argument('--output_prefix', help='Output file prefix (optional)')
    
    args = parser.parse_args()
    
    # Generate output prefix if not provided
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        input_path = Path(args.input_file)
        # Remove _32 from the end and replace with the subsample sizes
        stem = input_path.stem
        if stem.endswith('_32'):
            stem = stem[:-3]  # Remove the '_32'
        output_prefix = input_path.parent / stem
    
    process_subsample(args.input_file, output_prefix)

if __name__ == "__main__":
    main()