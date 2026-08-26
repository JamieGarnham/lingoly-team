#!/usr/bin/env python3
"""
Evaluation script for chained prompting responses.
Compatible with the output format of chained_prompting_rosetta.py
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any
import ast
import unicodedata

def normalize_answer(answer):
    """Normalize an answer for comparison with Unicode normalization"""
    if not isinstance(answer, str):
        answer = str(answer)
    # Normalize Unicode (NFC = canonical composition)
    answer = unicodedata.normalize('NFC', answer)
    return answer.strip().rstrip('.').lower()

def is_list_match(expected, actual):
    """Check if list answers match (handles string representations of lists)"""
    try:
        # Try to parse as literal if it's a string representation
        if isinstance(expected, str) and expected.startswith('['):
            expected = ast.literal_eval(expected)
        if isinstance(actual, str) and actual.startswith('['):
            actual = ast.literal_eval(actual)
        
        if isinstance(expected, list) and isinstance(actual, list):
            # Normalize and compare lists (with Unicode normalization)
            expected_norm = [normalize_answer(item) for item in expected]
            actual_norm = [normalize_answer(item) for item in actual]
            return set(expected_norm) == set(actual_norm)
    except:
        pass
    
    # Fall back to string comparison
    return normalize_answer(expected) == normalize_answer(actual)

def evaluate_response(expected_answer: Dict, model_parsed_response: Dict) -> Dict:
    """Evaluate a single response"""
    results = {}
    total_subqs = len(expected_answer)
    correct_subqs = 0
    
    for subq_key, expected_val in expected_answer.items():
        actual_val = model_parsed_response.get(subq_key, "")
        
        if is_list_match(expected_val, actual_val):
            is_correct = True
            correct_subqs += 1
        else:
            is_correct = False
        
        results[f"subq_{subq_key}_correct"] = is_correct
        results[f"subq_{subq_key}_expected"] = expected_val
        results[f"subq_{subq_key}_actual"] = actual_val
    
    results["total_subquestions"] = total_subqs
    results["correct_subquestions"] = correct_subqs
    results["accuracy"] = correct_subqs / total_subqs if total_subqs > 0 else 0.0
    
    return results

def load_and_evaluate(input_file: str) -> List[Dict]:
    """Load chained prompting responses and evaluate them"""
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                
                # Evaluate this response
                eval_result = evaluate_response(
                    data['expected_answer'], 
                    data['model_parsed_response']
                )
                
                # Combine with original data
                result = {
                    'overall_question_n': data['overall_question_n'],
                    'question_n': data['question_n'],
                    'obfuscated_question_n': data['obfuscated_question_n'],
                    'obf_num': data['obf_num'],
                    'split_key': data['split_key'],
                    'reasoning_length': len(data.get('reasoning_output', '')),
                    'response_length': len(data.get('model_raw_response', '')),
                    **eval_result
                }
                
                results.append(result)
    
    return results

def generate_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics"""
    if not results:
        return {}
    
    total_questions = len(results)
    total_subquestions = sum(r['total_subquestions'] for r in results)
    correct_subquestions = sum(r['correct_subquestions'] for r in results)
    perfect_questions = sum(1 for r in results if r['accuracy'] == 1.0)
    
    # By problem statistics
    by_problem = {}
    for result in results:
        prob_n = result['overall_question_n']
        if prob_n not in by_problem:
            by_problem[prob_n] = {'total': 0, 'correct_subqs': 0, 'total_subqs': 0}
        by_problem[prob_n]['total'] += 1
        by_problem[prob_n]['correct_subqs'] += result['correct_subquestions']
        by_problem[prob_n]['total_subqs'] += result['total_subquestions']
    
    return {
        'total_questions': total_questions,
        'total_subquestions': total_subquestions,
        'correct_subquestions': correct_subquestions,
        'perfect_questions': perfect_questions,
        'question_accuracy': perfect_questions / total_questions,
        'subquestion_accuracy': correct_subquestions / total_subquestions,
        'avg_reasoning_length': sum(r['reasoning_length'] for r in results) / total_questions,
        'avg_response_length': sum(r['response_length'] for r in results) / total_questions,
        'problems_evaluated': len(by_problem),
        'by_problem': by_problem
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate chained prompting responses')
    parser.add_argument('input_file', help='Input JSONL file with chained prompting responses')
    parser.add_argument('--output_csv', help='Output CSV file for detailed results')
    parser.add_argument('--summary_json', help='Output JSON file for summary statistics')
    
    args = parser.parse_args()
    
    # Load and evaluate
    print(f"Loading responses from {args.input_file}...")
    results = load_and_evaluate(args.input_file)
    print(f"Evaluated {len(results)} responses")
    
    # Generate summary
    summary = generate_summary(results)
    
    # Print summary to console
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Total Subquestions: {summary['total_subquestions']}")
    print(f"Perfect Questions: {summary['perfect_questions']} ({summary['question_accuracy']:.2%})")
    print(f"Correct Subquestions: {summary['correct_subquestions']} ({summary['subquestion_accuracy']:.2%})")
    print(f"Problems Evaluated: {summary['problems_evaluated']}")
    print(f"Avg Reasoning Length: {summary['avg_reasoning_length']:.0f} chars")
    print(f"Avg Response Length: {summary['avg_response_length']:.0f} chars")
    
    print(f"\nBy Problem:")
    for prob_n, stats in summary['by_problem'].items():
        accuracy = stats['correct_subqs'] / stats['total_subqs'] if stats['total_subqs'] > 0 else 0
        print(f"  Problem {prob_n}: {stats['correct_subqs']}/{stats['total_subqs']} ({accuracy:.2%})")
    
    # Save detailed results to CSV
    if args.output_csv:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            if results:
                # Get all possible field names from all results
                all_fieldnames = set()
                for result in results:
                    all_fieldnames.update(result.keys())
                
                # Sort fieldnames for consistent ordering
                fieldnames = sorted(all_fieldnames)
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
        print(f"\nDetailed results saved to {args.output_csv}")
    
    # Save summary to JSON
    if args.summary_json:
        with open(args.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to {args.summary_json}")

if __name__ == "__main__":
    main()