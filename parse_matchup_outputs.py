#!/usr/bin/env python3
"""
Script to parse match-up evaluation outputs and extract final answers.

Takes JSONL files from the chained prompting match-up evaluation and extracts
the final answer for each question after all match-up rounds are complete.

Output format: {"a": "answer", "b": "answer", ...} where a, b, etc. are the 
serials/question part numbers and the values are their matched answers.

Rules:
- If a serial is matched multiple times, take the first match in the sequence
- If a serial is not matched, leave its answer blank
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any
import sys


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


def extract_final_answers(question_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Extract final answers for each question from the match-up data.
    
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
            # Sort by match_round to process in order
            entries.sort(key=lambda x: x.get('match_round', 0))
            
            # Get all question parts from the question_details
            if not entries:
                continue
                
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
            
            # Process each match round to build up confirmed matches
            seen_matches = set()  # Track which question parts have been matched (using expected format)
            
            for entry in entries:
                model_response = entry.get('model_parsed_response', {})
                
                # Process matches from this round
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


def main():
    parser = argparse.ArgumentParser(description='Parse match-up evaluation outputs and extract final answers')
    parser.add_argument('files', nargs='+', help='JSONL files to process')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    parser.add_argument('--pretty', '-p', action='store_true', help='Pretty print JSON output')
    
    args = parser.parse_args()
    
    all_results = []
    
    for file_path in args.files:
        print(f"Processing {file_path}...", file=sys.stderr)
        results = parse_matchup_file(file_path)
        if results:
            all_results.extend(results)
            print(f"  Found {len(results)} entries", file=sys.stderr)
        else:
            print(f"  No valid entries found", file=sys.stderr)
    
    if not all_results:
        print("No data to process", file=sys.stderr)
        return 1
    
    # Extract final answers
    final_answers = extract_final_answers(all_results)
    
    # Prepare output
    total_sub_questions = sum(len(sub_questions) for sub_questions in final_answers.values())
    output_data = {
        'summary': {
            'total_overall_questions': len(final_answers),
            'total_sub_questions': total_sub_questions,
            'files_processed': args.files
        },
        'final_answers': final_answers
    }
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.pretty:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(output_data, f, ensure_ascii=False)
    else:
        if args.pretty:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(output_data, ensure_ascii=False))
    
    # Print summary to stderr
    total_sub_questions = sum(len(sub_questions) for sub_questions in final_answers.values())
    print(f"\nProcessed {len(final_answers)} overall questions ({total_sub_questions} sub-questions) from {len(args.files)} files", file=sys.stderr)
    for overall_question_id, sub_questions in final_answers.items():
        print(f"  {overall_question_id}:", file=sys.stderr)
        for sub_question_id, answers in sub_questions.items():
            filled_count = sum(1 for v in answers.values() if v.strip())
            total_count = len(answers)
            print(f"    {sub_question_id}: {filled_count}/{total_count} parts answered", file=sys.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())