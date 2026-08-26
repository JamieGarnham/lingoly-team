#!/usr/bin/env python3
"""
Simple script to parse match-up evaluation outputs and extract final confirmed matches.

For each matchup round question in the .jsonl input file, this script takes the list 
of confirmed_matches from the final match_round for that question and converts it to a dict.

The output dict may contain duplicate keys or missing keys - this is intentional and
reflects the raw confirmed_matches data.

For follow-up questions (non-matchup rounds), these are parsed perfectly by the original
script, so they are not modified.
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any
import sys
import csv
import os
import re


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
                    print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}", file=sys.stderr)
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return []
    
    return results


def is_matchup_question(entry: Dict[str, Any]) -> bool:
    """Check if this entry represents a matchup round question."""
    return 'match_round' in entry and entry.get('match_round') is not None


def extract_simple_answers(question_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Extract simple answers for each question from the match-up data.
    
    For matchup questions: takes confirmed_matches from the final match_round and converts to dict
    For non-matchup questions: uses the existing logic from the original script
    
    Returns a dict mapping overall_question_id -> sub_question_id -> final_answer_dict
    """
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
                
                # Sort by match_round to find the final round
                entries.sort(key=lambda x: x.get('match_round', 0))
                final_entry = entries[-1]  # Last entry has the highest match_round
                
                confirmed_matches = final_entry.get('confirmed_matches', [])
                
                # Convert confirmed_matches list to dict
                # confirmed_matches format: [["f", "4"], ["f", "4"], ["e", "2"], ...]
                final_answer = {}
                for match_pair in confirmed_matches:
                    if len(match_pair) >= 2:
                        key, value = match_pair[0], match_pair[1]
                        # Note: This will overwrite duplicate keys, keeping the last occurrence
                        # If you want to preserve all duplicates, you'd need a different data structure
                        final_answer[key] = value
                
                final_answers[overall_question_id][sub_question_id] = final_answer
                
            else:
                # Handle non-matchup questions using the original logic
                # This is essentially the same as the original script for follow-up questions
                
                # Sort by match_round if available (though these shouldn't have match_round)
                entries.sort(key=lambda x: x.get('match_round', 0))
                
                # Get all question parts from the question_details
                question_details = entries[0].get('question_details', {})
                
                # Handle both single question and multi-question formats
                if 'subprompts' in question_details:
                    # Single question format
                    subprompts = question_details.get('subprompts', [])
                else:
                    # Multi-question format - find the matching sub-question
                    questions_list = question_details.get('questions', [])
                    if isinstance(questions_list, str):
                        # Parse JSON string if needed
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
                    """Normalize part ID by removing dots, brackets, and converting to lowercase"""
                    if not isinstance(part_id, str):
                        part_id = str(part_id)
                    return part_id.lower().rstrip('.').rstrip(':').strip('()[]{}')
                
                # Create mapping from normalized to expected format
                normalized_to_expected = {}
                for expected_part in expected_parts:
                    normalized = normalize_part_id(expected_part)
                    normalized_to_expected[normalized] = expected_part
                
                # Process each entry to build up confirmed matches
                seen_matches = set()  # Track which question parts have been matched (using expected format)
                
                for entry in entries:
                    model_response = entry.get('model_parsed_response', {})
                    
                    # Process matches from this entry
                    for question_part, answer in model_response.items():
                        # Normalize the model's question part identifier
                        normalized_part = normalize_part_id(question_part)
                        
                        # Find the corresponding expected format
                        if normalized_part in normalized_to_expected:
                            expected_part = normalized_to_expected[normalized_part]
                            
                            # Only take the first match for each question part
                            if expected_part not in seen_matches:
                                final_answer[expected_part] = answer
                                seen_matches.add(expected_part)
                
                final_answers[overall_question_id][sub_question_id] = final_answer
    
    return final_answers


def extract_model_name_and_timestamp(filename: str) -> tuple[str, str]:
    """Extract model name and timestamp from filename."""
    # Remove path and extension
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    
    # Pattern to match model_name_matchup_chained_evaluation_timestamp
    # Example: Gemini_2.5_Flash_matchup_chained_evaluation_20251017_132122
    pattern = r'^(.+?)_matchup_chained_evaluation_(\d{8}_\d{6})$'
    match = re.match(pattern, basename)
    
    if match:
        model_name = match.group(1)
        timestamp = match.group(2)
        return model_name, timestamp
    else:
        # Fallback: try to extract any timestamp at the end
        timestamp_pattern = r'_(\d{8}_\d{6})$'
        timestamp_match = re.search(timestamp_pattern, basename)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            model_name = basename[:timestamp_match.start()]
        else:
            model_name = basename
            timestamp = "unknown"
        return model_name, timestamp


def determine_output_folder(model_name: str) -> str:
    """Determine the output folder based on model name."""
    model_name_lower = model_name.lower()
    
    if 'gemini' in model_name_lower:
        return 'openrouter_runs/gemini_chained_matchup'
    elif 'r1' in model_name_lower or 'deepseek' in model_name_lower:
        return 'openrouter_runs/deepseek_chained_matchup'
    elif 'llama' in model_name_lower:
        return 'openrouter_runs/llama_chained_matchup'
    else:
        # Default fallback
        return 'openrouter_runs/other_chained_matchup'


def clean_overall_question_n(question_id: str) -> str:
    """Remove the obfuscation number suffix from overall_question_n.
    
    Examples:
        "167_0004" -> "167"
        "127_0005" -> "127"
        "74_0006" -> "74"
    """
    if '_' in question_id:
        return question_id.split('_')[0]
    return question_id


def generate_csv_filename(input_file: str, model_name: str, timestamp: str) -> str:
    """Generate CSV output filename based on input file, model name and timestamp."""
    folder = determine_output_folder(model_name)
    
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    filename = f"{model_name}_matchup_outputs_{timestamp}.csv"
    return os.path.join(folder, filename)


def main():
    parser = argparse.ArgumentParser(description='Parse match-up evaluation outputs and extract simple final answers as CSV')
    parser.add_argument('files', nargs='+', help='JSONL files to process')
    parser.add_argument('--output', '-o', help='Output CSV file (default: auto-generated from input filename)')
    
    args = parser.parse_args()
    
    all_results = []
    model_name = None
    timestamp = None
    
    for file_path in args.files:
        print(f"Processing {file_path}...", file=sys.stderr)
        results = parse_matchup_file(file_path)
        if results:
            all_results.extend(results)
            print(f"  Found {len(results)} entries", file=sys.stderr)
            
            # Extract model name and timestamp from first file
            if model_name is None:
                model_name, timestamp = extract_model_name_and_timestamp(file_path)
                print(f"  Extracted model: {model_name}, timestamp: {timestamp}", file=sys.stderr)
        else:
            print(f"  No valid entries found", file=sys.stderr)
    
    if not all_results:
        print("No data to process", file=sys.stderr)
        return 1
    
    # Extract simple answers
    final_answers = extract_simple_answers(all_results)
    
    # Prepare CSV data
    csv_rows = []
    for overall_question_id, sub_questions in final_answers.items():
        for sub_question_id, answers in sub_questions.items():
            # Convert answers dict to JSON string for the model_answers column
            model_answers_json = json.dumps(answers, ensure_ascii=False)
            # Clean the overall_question_n to remove obfuscation suffix
            clean_question_id = clean_overall_question_n(overall_question_id)
            csv_rows.append({
                'overall_question_n': clean_question_id,
                'question_n': sub_question_id,
                'model_answers': model_answers_json
            })
    
    # Determine output filename
    if args.output:
        csv_filename = args.output
    else:
        csv_filename = generate_csv_filename(args.files[0], model_name or "unknown", timestamp or "unknown")
    
    # Write CSV with UTF-8 BOM for proper Google Sheets compatibility
    with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['overall_question_n', 'question_n', 'model_answers']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    
    print(f"\nCSV output written to: {csv_filename}", file=sys.stderr)
    print(f"Processed {len(final_answers)} overall questions ({len(csv_rows)} sub-questions) from {len(args.files)} files", file=sys.stderr)
    
    # Print summary to stderr
    for overall_question_id, sub_questions in final_answers.items():
        print(f"  {overall_question_id}:", file=sys.stderr)
        for sub_question_id, answers in sub_questions.items():
            if isinstance(answers, dict):
                filled_count = sum(1 for v in answers.values() if str(v).strip())
                total_count = len(answers)
                print(f"    {sub_question_id}: {filled_count}/{total_count} parts answered", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())