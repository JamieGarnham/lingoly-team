#!/usr/bin/env python3
"""
Script to analyze question formats in benchmark_same_obf.jsonl and count multiple choice vs non-multiple choice questions.
"""

import json
import csv
from collections import Counter
from pathlib import Path

def load_exam_papers_format():
    """Load the past-exam-papers.csv to get question formats."""
    exam_papers_path = Path("testing/data/past-exam-papers.csv")
    if not exam_papers_path.exists():
        print(f"Warning: {exam_papers_path} not found.")
        return {}
    
    format_mapping = {}
    with open(exam_papers_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            overall_q_num = int(row['Overall Question Number'])
            question_format = row['Question Format']
            # Parse the format field which looks like "['Match-up']" or "['Rosetta']"
            if question_format.startswith('[') and question_format.endswith(']'):
                # Remove brackets and quotes
                format_clean = question_format.strip("[]'\"")
                format_mapping[overall_q_num] = format_clean
            else:
                format_mapping[overall_q_num] = question_format
    
    return format_mapping

def is_multiple_choice_format(format_str):
    """Determine if a question format is multiple choice."""
    # Based on the sample data, "Match-up" appears to be multiple choice
    # while "Rosetta", "Pattern", "Text" are not
    multiple_choice_formats = ["Match-up"]
    return format_str in multiple_choice_formats

def analyze_benchmark_questions():
    """Analyze the benchmark file to count question formats."""
    benchmark_path = Path("testing/data/splits/benchmark_same_obf.jsonl")
    if not benchmark_path.exists():
        print(f"Error: {benchmark_path} not found.")
        return
    
    # Load format mapping
    format_mapping = load_exam_papers_format()
    
    # Track questions by overall question number to avoid double counting
    questions_seen = set()
    format_counts = Counter()
    multiple_choice_count = 0
    non_multiple_choice_count = 0
    unknown_format_count = 0
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                overall_question_n = data['index'][0]
                
                # Only count each overall question once
                if overall_question_n not in questions_seen:
                    questions_seen.add(overall_question_n)
                    
                    # Get format from mapping
                    question_format = format_mapping.get(overall_question_n, "Unknown")
                    format_counts[question_format] += 1
                    
                    # Categorize as multiple choice or not
                    if question_format == "Unknown":
                        unknown_format_count += 1
                    elif is_multiple_choice_format(question_format):
                        multiple_choice_count += 1
                    else:
                        non_multiple_choice_count += 1
                        
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line: {line.strip()[:100]}...")
                continue
    
    # Print results
    print("Question Format Analysis:")
    print("=" * 40)
    print(f"Total unique questions: {len(questions_seen)}")
    print(f"Multiple choice questions: {multiple_choice_count}")
    print(f"Non-multiple choice questions: {non_multiple_choice_count}")
    print(f"Unknown format questions: {unknown_format_count}")
    print()
    print("Breakdown by format:")
    for format_type, count in format_counts.most_common():
        mc_indicator = " (Multiple Choice)" if is_multiple_choice_format(format_type) else ""
        print(f"  {format_type}: {count}{mc_indicator}")
    
    return {
        'total': len(questions_seen),
        'multiple_choice': multiple_choice_count,
        'non_multiple_choice': non_multiple_choice_count,
        'unknown': unknown_format_count,
        'format_breakdown': dict(format_counts)
    }

if __name__ == "__main__":
    analyze_benchmark_questions()