#!/usr/bin/env python3
"""
Script to evaluate multiple runs of matchup chained prompting at the subquestion level.

This script:
1. Finds JSONL files for a given model in testing/data/chained_responses/
2. Copies them to openrouter_runs/[model]_chained_matchup/ with version prefixes
3. Parses answers from each version using matchup parsing logic
4. Evaluates answers at subquestion level with majority voting and correctness analysis
5. Outputs comprehensive CSV with per-version and aggregated results
"""

import json
import csv
import os
import sys
import argparse
import shutil
import re
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
from datetime import datetime


def normalize_answer(answer):
    """Normalize an answer for comparison."""
    if not isinstance(answer, str):
        answer = str(answer)
    return answer.strip().rstrip('.').lower()


def is_valid_model_answer(answer):
    """Check if a model answer is valid (not empty, N/A, or Invalid JSON)."""
    if not answer or answer == "N/A":
        return False
    
    if not isinstance(answer, str):
        answer = str(answer)
    
    answer_lower = answer.lower().strip()
    if "invalid json" in answer_lower or answer_lower == "":
        return False
    return True


def calculate_majority_and_tiebreaker(model_answers):
    """
    Calculate majority answer and tiebreaker from a list of model answers.
    Returns: (majority, tiebreaker, majority_size, unique_answers)
    """
    # Set fixed random seed for reproducible results
    random.seed(42)
    
    # Filter out empty/invalid answers
    valid_answers = [ans for ans in model_answers if is_valid_model_answer(ans)]
    
    if not valid_answers:
        return "", "", 0, []
    
    # Count occurrences
    answer_counts = Counter(valid_answers)
    unique_answers = list(answer_counts.keys())
    
    # Find most common answer(s)
    max_count = max(answer_counts.values())
    most_common_answers = [ans for ans, count in answer_counts.items() if count == max_count]
    
    # Determine majority
    if len(most_common_answers) == 1:
        majority = most_common_answers[0]
        tiebreaker = majority
    else:
        majority = "N/A"  # No clear majority
        tiebreaker = random.choice(most_common_answers)  # Random tiebreaker
    
    return majority, tiebreaker, max_count, unique_answers


def get_model_file_pattern(model_name: str) -> str:
    """Get the filename pattern for the specified model."""
    patterns = {
        "gemini": "Gemini_2.5_Flash",
        "deepseek": "R1", 
        "llama": "Llama_3.3_70B_OpenRouter"
    }
    
    if model_name not in patterns:
        raise ValueError(f"Unknown model: {model_name}. Must be one of: {list(patterns.keys())}")
    
    return patterns[model_name]


def find_model_files(input_dir: str, model_name: str) -> List[str]:
    """Find all JSONL files for the specified model, sorted by timestamp."""
    pattern = get_model_file_pattern(model_name)
    files = []
    
    for filename in os.listdir(input_dir):
        if (filename.startswith(pattern) and 
            '_matchup_chained_evaluation_' in filename and 
            filename.endswith('.jsonl')):
            files.append(os.path.join(input_dir, filename))
    
    # Sort by timestamp (extracted from filename)
    def extract_timestamp(filepath):
        basename = os.path.basename(filepath)
        # Extract timestamp like 20251017_132122
        match = re.search(r'(\d{8}_\d{6})', basename)
        return match.group(1) if match else basename
    
    files.sort(key=extract_timestamp)
    return files


def copy_and_version_files(source_files: List[str], output_dir: str, num_versions: int) -> List[str]:
    """Copy files to output directory with version prefixes."""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(source_files) < num_versions:
        raise ValueError(f"Requested {num_versions} versions but only {len(source_files)} files found")
    
    versioned_files = []
    for i in range(num_versions):
        source_file = source_files[i]
        basename = os.path.basename(source_file)
        versioned_name = f"v{i+1}_{basename}"
        dest_path = os.path.join(output_dir, versioned_name)
        
        shutil.copy2(source_file, dest_path)
        versioned_files.append(dest_path)
        print(f"Copied {basename} -> {versioned_name}")
    
    return versioned_files


def parse_matchup_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a JSONL file and return list of question results."""
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    results.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
    
    return results


def is_matchup_question(entry: Dict[str, Any]) -> bool:
    """Check if this entry represents a matchup round question."""
    return 'match_round' in entry and entry.get('match_round') is not None


def clean_overall_question_n(question_id: str) -> str:
    """Remove the obfuscation number suffix from overall_question_n."""
    if '_' in question_id:
        return question_id.split('_')[0]
    return question_id


def extract_answers_from_file(file_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Extract answers from a single JSONL file using matchup parsing logic.
    Returns dict: overall_question_id -> sub_question_id -> {subquestion: answer}
    """
    question_data = parse_matchup_file(file_path)
    if not question_data:
        return {}
    
    # Group by overall question and sub-question
    questions = defaultdict(lambda: defaultdict(list))
    for entry in question_data:
        overall_question_id = entry.get('obfuscated_question_n', entry.get('overall_question_n', 'unknown'))
        sub_question_id = entry.get('question_n', 'main')
        questions[overall_question_id][sub_question_id].append(entry)
    
    final_answers = {}
    
    for overall_question_id, sub_questions in questions.items():
        final_answers[overall_question_id] = {}
        
        for sub_question_id, entries in sub_questions.items():
            if not entries:
                continue
            
            # Check if this is a matchup question
            if is_matchup_question(entries[0]):
                # Handle matchup questions: find the final match_round and extract confirmed_matches
                entries.sort(key=lambda x: x.get('match_round', 0))
                final_entry = entries[-1]  # Last entry has the highest match_round
                
                confirmed_matches = final_entry.get('confirmed_matches', [])
                
                # Convert confirmed_matches list to dict
                final_answer = {}
                for match_pair in confirmed_matches:
                    if len(match_pair) >= 2:
                        key, value = match_pair[0], match_pair[1]
                        final_answer[key] = value
                
                final_answers[overall_question_id][sub_question_id] = final_answer
                
            else:
                # Handle non-matchup questions using the original logic
                entries.sort(key=lambda x: x.get('match_round', 0))
                
                # Get all question parts from the question_details
                question_details = entries[0].get('question_details', {})
                
                # Handle both single question and multi-question formats
                if 'subprompts' in question_details:
                    subprompts = question_details.get('subprompts', [])
                else:
                    questions_list = question_details.get('questions', [])
                    if isinstance(questions_list, str):
                        try:
                            questions_list = json.loads(questions_list)
                        except:
                            questions_list = []
                    
                    subprompts = []
                    for q in questions_list:
                        if q.get('question_n') == sub_question_id:
                            subprompts = q.get('subprompts', [])
                            break
                
                # Initialize answer dict with empty strings for all question parts
                final_answer = {}
                expected_parts = []
                for subprompt in subprompts:
                    part_n = subprompt.get('questionpart_n', '')
                    expected_parts.append(part_n)
                    final_answer[part_n] = ""
                
                # Create a mapping function to normalize question part identifiers
                def normalize_part_id(part_id):
                    if not isinstance(part_id, str):
                        part_id = str(part_id)
                    return part_id.lower().rstrip('.').rstrip(':').strip('()[]{}')
                
                # Create mapping from normalized to expected format
                normalized_to_expected = {}
                for expected_part in expected_parts:
                    normalized = normalize_part_id(expected_part)
                    normalized_to_expected[normalized] = expected_part
                
                # Process each entry to build up confirmed matches
                seen_matches = set()
                
                for entry in entries:
                    model_response = entry.get('model_parsed_response', {})
                    
                    for question_part, answer in model_response.items():
                        normalized_part = normalize_part_id(question_part)
                        
                        if normalized_part in normalized_to_expected:
                            expected_part = normalized_to_expected[normalized_part]
                            
                            if expected_part not in seen_matches:
                                final_answer[expected_part] = answer
                                seen_matches.add(expected_part)
                
                final_answers[overall_question_id][sub_question_id] = final_answer
    
    return final_answers


def get_question_context(entries: List[Dict[str, Any]]) -> str:
    """Extract the full problem context/preamble from question entries."""
    if not entries:
        return ""
    
    question_details = entries[0].get('question_details', {})
    metadata = question_details.get('metadata', {})
    
    preamble = metadata.get('preamble', '')
    context = metadata.get('context', '')
    
    # Combine preamble and context
    full_context = ""
    if preamble:
        full_context += preamble.strip()
    if context:
        if full_context:
            full_context += "\n\n"
        full_context += context.strip()
    
    return full_context


def get_expected_answers(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract expected answers for all subquestions from question entries."""
    if not entries:
        return {}
    
    # Get expected answer from the first entry
    expected_answer = entries[0].get('expected_answer', {})
    return expected_answer


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate multiple runs of matchup chained prompting at subquestion level'
    )
    parser.add_argument('model', choices=['gemini', 'deepseek', 'llama'],
                        help='Model name to evaluate')
    parser.add_argument('num_versions', type=int,
                        help='Number of JSONL files to use (X parameter)')
    parser.add_argument('--input_dir', default='testing/data/chained_responses',
                        help='Input directory containing JSONL files')
    
    args = parser.parse_args()
    
    # Find model files
    print(f"Looking for {args.model} files in {args.input_dir}...")
    model_files = find_model_files(args.input_dir, args.model)
    
    if not model_files:
        print(f"No files found for model {args.model}")
        return 1
    
    print(f"Found {len(model_files)} files for {args.model}")
    
    if args.num_versions > len(model_files):
        print(f"Error: Requested {args.num_versions} versions but only {len(model_files)} files available")
        return 1
    
    # Copy and version files
    output_dir = f"openrouter_runs/{args.model}_chained_matchup"
    versioned_files = copy_and_version_files(model_files, output_dir, args.num_versions)
    
    # Extract answers from each version
    print(f"\nExtracting answers from {args.num_versions} versions...")
    all_version_answers = []
    all_questions_data = {}  # Store question context and expected answers
    
    for i, file_path in enumerate(versioned_files):
        print(f"Processing version {i+1}...")
        version_answers = extract_answers_from_file(file_path)
        all_version_answers.append(version_answers)
        
        # Also collect question data from first file
        if i == 0:
            question_data = parse_matchup_file(file_path)
            questions = defaultdict(lambda: defaultdict(list))
            for entry in question_data:
                overall_question_id = entry.get('obfuscated_question_n', entry.get('overall_question_n', 'unknown'))
                sub_question_id = entry.get('question_n', 'main')
                questions[overall_question_id][sub_question_id].append(entry)
            
            for overall_question_id, sub_questions in questions.items():
                all_questions_data[overall_question_id] = {}
                for sub_question_id, entries in sub_questions.items():
                    all_questions_data[overall_question_id][sub_question_id] = {
                        'context': get_question_context(entries),
                        'expected_answers': get_expected_answers(entries)
                    }
    
    # Prepare CSV data
    print("\nPreparing CSV data...")
    csv_rows = []
    
    for overall_question_id, sub_questions in all_questions_data.items():
        clean_question_id = clean_overall_question_n(overall_question_id)
        
        for sub_question_id, question_info in sub_questions.items():
            context = question_info['context']
            expected_answers = question_info['expected_answers']
            
            # For each subquestion in expected answers
            for serial, correct_answer in expected_answers.items():
                # Collect model answers across all versions
                model_answers = []
                for version_answers in all_version_answers:
                    version_answer = ""
                    if (overall_question_id in version_answers and 
                        sub_question_id in version_answers[overall_question_id]):
                        # Try exact match first
                        version_answer = version_answers[overall_question_id][sub_question_id].get(serial, "")
                        
                        # If no exact match, try normalized matching
                        if not version_answer:
                            # Normalize the expected key by removing periods and other punctuation
                            normalized_serial = serial.rstrip('.').rstrip(':').strip('()[]{}')
                            version_answer = version_answers[overall_question_id][sub_question_id].get(normalized_serial, "")
                            
                            # Also try the reverse - if expected key has no punctuation, try adding period
                            if not version_answer and not any(c in serial for c in '.:()'): 
                                version_answer = version_answers[overall_question_id][sub_question_id].get(serial + '.', "")
                    
                    model_answers.append(version_answer)
                
                # Calculate majority and statistics
                majority, tiebreaker, majority_size, unique_answers = calculate_majority_and_tiebreaker(model_answers)
                
                # Count correct answers
                number_correct = sum(1 for ans in model_answers 
                                   if is_valid_model_answer(ans) and normalize_answer(ans) == normalize_answer(correct_answer))
                
                # Check if any answer is correct
                is_any_correct = number_correct > 0
                
                # Check if majority/tiebreaker are correct
                is_majority_correct = ""
                is_tiebreaker_correct = ""
                if majority and is_valid_model_answer(majority):
                    is_majority_correct = normalize_answer(majority) == normalize_answer(correct_answer)
                if tiebreaker and is_valid_model_answer(tiebreaker):
                    is_tiebreaker_correct = normalize_answer(tiebreaker) == normalize_answer(correct_answer)
                
                # Create row
                row = {
                    'questions': context,
                    'overall_question_n': clean_question_id,
                    'question_n': sub_question_id,
                    'serial': serial,
                    'format': "['Match-up']",
                    'correct_answer': correct_answer,
                    'majority': majority,
                    'tiebreaker': tiebreaker,
                    'majority_size': majority_size,
                    'unique_answers': str(unique_answers),
                    'is_majority_correct': is_majority_correct,
                    'is_tiebreaker_correct': is_tiebreaker_correct,
                    'is_any_correct': is_any_correct,
                    'number_correct': number_correct,
                    'format_fixed': 'Match-up'
                }
                
                # Add individual version columns
                for i, model_answer in enumerate(model_answers):
                    version_num = i + 1
                    row[f'model_answer_v{version_num}'] = model_answer
                    
                    is_correct = ""
                    if is_valid_model_answer(model_answer):
                        is_correct = normalize_answer(model_answer) == normalize_answer(correct_answer)
                    row[f'is_v{version_num}_correct'] = is_correct
                
                csv_rows.append(row)
    
    # Write CSV
    output_analysis_dir = f"openrouter_analysis/{args.model}"
    os.makedirs(output_analysis_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{args.model}_matchup_v{args.num_versions}_evaluation_{timestamp}.csv"
    csv_path = os.path.join(output_analysis_dir, csv_filename)
    
    if csv_rows:
        # Determine column order
        base_columns = [
            'questions', 'overall_question_n', 'question_n', 'serial', 'format', 'correct_answer'
        ]
        
        # Add version columns
        version_columns = []
        for i in range(args.num_versions):
            version_num = i + 1
            version_columns.extend([f'model_answer_v{version_num}', f'is_v{version_num}_correct'])
        
        final_columns = [
            'majority', 'tiebreaker', 'majority_size', 'unique_answers',
            'is_majority_correct', 'is_tiebreaker_correct', 'is_any_correct', 
            'number_correct', 'format_fixed'
        ]
        
        all_columns = base_columns + version_columns + final_columns
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\nCSV written to: {csv_path}")
        print(f"Processed {len(csv_rows)} subquestions across {len(all_questions_data)} questions")
    else:
        print("No data to write to CSV")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())