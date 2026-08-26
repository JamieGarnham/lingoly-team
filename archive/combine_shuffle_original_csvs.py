#!/usr/bin/env python3
"""
Script to combine shuffle and original CSV files:
- If a question (overall_question_n) is in the shuffle CSV, use the shuffle row
- Otherwise, use the row from the original CSV
- Output as a new combined CSV
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict

# Increase CSV field size limit to handle large question text
import sys
csv.field_size_limit(sys.maxsize)

def load_csv_by_question(csv_path):
    """
    Load CSV and organize rows by overall_question_n for easy lookup.
    Returns: dict mapping overall_question_n -> list of rows for that question
    """
    questions_data = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            overall_question_n = int(row['overall_question_n'])
            questions_data[overall_question_n].append(row)
    
    return questions_data

def combine_csvs(shuffle_csv_path, original_csv_path, output_csv_path):
    """
    Combine shuffle and original CSVs according to the priority rules.
    """
    print("Loading shuffle CSV...")
    shuffle_data = load_csv_by_question(shuffle_csv_path)
    shuffle_questions = set(shuffle_data.keys())
    print(f"Shuffle CSV contains {len(shuffle_questions)} unique questions: {sorted(shuffle_questions)}")
    
    print("Loading original CSV...")
    original_data = load_csv_by_question(original_csv_path)
    original_questions = set(original_data.keys())
    print(f"Original CSV contains {len(original_questions)} unique questions")
    
    # Create combined data
    combined_rows = []
    
    # First, add all questions from shuffle CSV (these take priority)
    for question_n in sorted(shuffle_questions):
        combined_rows.extend(shuffle_data[question_n])
        print(f"Added {len(shuffle_data[question_n])} rows from shuffle for question {question_n}")
    
    # Then, add questions from original CSV that are NOT in shuffle CSV
    original_only_questions = original_questions - shuffle_questions
    for question_n in sorted(original_only_questions):
        combined_rows.extend(original_data[question_n])
        print(f"Added {len(original_data[question_n])} rows from original for question {question_n}")
    
    print(f"\nCombined data statistics:")
    print(f"Total rows: {len(combined_rows)}")
    print(f"Questions from shuffle: {len(shuffle_questions)}")
    print(f"Questions from original only: {len(original_only_questions)}")
    print(f"Total unique questions: {len(shuffle_questions) + len(original_only_questions)}")
    
    # Write combined CSV
    if combined_rows:
        fieldnames = combined_rows[0].keys()
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        
        print(f"\nSaved combined CSV to: {output_csv_path}")
        print(f"Total rows written: {len(combined_rows)} (plus header)")
    else:
        print("No data to write!")

def main():
    """
    Main function to combine shuffle and original CSV files.
    """
    parser = argparse.ArgumentParser(description='Combine shuffle and original CSV files')
    parser.add_argument('--shuffle-csv', 
                       default='/Users/jamiegarnham/lingoly2/openrouter_analysis/subquestion_eval_deepseek_shuffle_with_correctness_16.csv',
                       help='Path to shuffle CSV file')
    parser.add_argument('--original-csv',
                       default='/Users/jamiegarnham/lingoly2/openrouter_analysis/subquestion_eval_original_prompt_with_correctness_16.csv', 
                       help='Path to original CSV file')
    parser.add_argument('--output-csv',
                       default='/Users/jamiegarnham/lingoly2/openrouter_analysis/subquestion_eval_combined_shuffle_original_16.csv',
                       help='Path for output combined CSV file')
    
    args = parser.parse_args()
    
    # Verify input files exist
    shuffle_path = Path(args.shuffle_csv)
    original_path = Path(args.original_csv)
    
    if not shuffle_path.exists():
        raise FileNotFoundError(f"Shuffle CSV not found: {shuffle_path}")
    if not original_path.exists():
        raise FileNotFoundError(f"Original CSV not found: {original_path}")
    
    print("=== Combining Shuffle and Original CSV Files ===")
    print(f"Shuffle CSV: {shuffle_path}")
    print(f"Original CSV: {original_path}")
    print(f"Output CSV: {args.output_csv}")
    print()
    
    combine_csvs(args.shuffle_csv, args.original_csv, args.output_csv)
    
    print("\n=== Combination Complete ===")

if __name__ == "__main__":
    main()