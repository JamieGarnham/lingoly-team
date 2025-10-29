#!/usr/bin/env python3
"""
Script to combine deepseek_shuffle and original_prompt results.
Uses deepseek_shuffle data where available, falls back to original_prompt for missing questions.
"""

import csv
import sys

def combine_datasets(deepseek_file, original_file, output_file):
    """
    Combine two CSV files, prioritizing deepseek_shuffle data.
    """
    # Increase CSV field size limit
    csv.field_size_limit(sys.maxsize)
    
    print("Loading deepseek_shuffle data...")
    deepseek_data = {}
    with open(deepseek_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create unique key for each question
            key = (row['overall_question_n'], row['question_n'], row['serial'])
            deepseek_data[key] = row
    
    print(f"Loaded {len(deepseek_data)} rows from deepseek_shuffle")
    
    print("Loading original_prompt data...")
    original_data = {}
    header = None
    with open(original_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            # Create unique key for each question
            key = (row['overall_question_n'], row['question_n'], row['serial'])
            original_data[key] = row
    
    print(f"Loaded {len(original_data)} rows from original_prompt")
    
    # Combine data: use deepseek where available, original as fallback
    print("Combining datasets...")
    combined_data = {}
    
    # Start with all original data
    combined_data.update(original_data)
    
    # Override with deepseek data where available
    for key, deepseek_row in deepseek_data.items():
        combined_data[key] = deepseek_row
    
    print(f"Combined dataset has {len(combined_data)} rows")
    print(f"  - {len(deepseek_data)} from deepseek_shuffle")
    print(f"  - {len(combined_data) - len(deepseek_data)} from original_prompt")
    
    # Sort by question order for consistent output
    sorted_keys = sorted(combined_data.keys(), 
                        key=lambda x: (int(x[0]), x[1], x[2]))
    
    # Write combined data
    print(f"Writing combined data to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for key in sorted_keys:
            writer.writerow(combined_data[key])
    
    print(f"Successfully created {output_file} with {len(combined_data)} rows")

if __name__ == "__main__":
    deepseek_file = "openrouter_analysis/subquestion_eval_deepseek_shuffle_32.csv"
    original_file = "openrouter_analysis/subquestion_eval_original_prompt_32.csv"
    output_file = "openrouter_analysis/subquestion_eval_combined_shuffle_original_32.csv"
    
    combine_datasets(deepseek_file, original_file, output_file)