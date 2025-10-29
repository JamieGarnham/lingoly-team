#!/usr/bin/env python3
"""
Script to extract reasoning outputs from chained prompt responses.
Creates a CSV where each row corresponds to a whole question with all 16 reasoning outputs.
"""

import json
import csv
import os
import sys
import argparse
from pathlib import Path
import re

def load_chained_prompt_responses(chained_prompts_dir, sample_size=None):
    """
    Load chained prompt response files from the specified directory.
    
    Args:
        chained_prompts_dir: Path to the chained_prompts directory
        sample_size: If specified, randomly sample this many files
    """
    chained_dir = Path(chained_prompts_dir)
    if not chained_dir.exists():
        raise FileNotFoundError(f"Directory {chained_dir} does not exist")
    
    model_files = []
    
    # Find all files that match v*_*evaluation*.jsonl pattern
    for file_path in chained_dir.glob("v*_*evaluation*.jsonl"):
        match = re.match(r'v(\d+)_.*evaluation.*\.jsonl', file_path.name)
        if match:
            version_num = int(match.group(1))
            model_files.append((version_num, file_path))
    
    if not model_files:
        raise FileNotFoundError(f"No matching JSONL files found in {chained_dir}")
    
    # Sort by version number
    model_files.sort(key=lambda x: x[0])
    
    # Apply sampling if requested
    if sample_size is not None and sample_size < len(model_files):
        # Take first N files after sorting
        model_files = model_files[:sample_size]
    
    responses_by_version = {}
    for version_num, file_path in model_files:
        version_key = f"v{version_num}"
        responses_by_version[version_key] = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    responses_by_version[version_key].append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse line in {file_path}")
                    continue
    
    return responses_by_version

def extract_question_reasoning_data(responses_by_version):
    """
    Extract reasoning data for each whole question.
    """
    rows = []
    
    # Group by (overall_question_n, question_n) to identify unique questions
    question_data = {}  # Maps (overall_question_n, question_n) -> {version -> response}
    
    for version_key, responses in responses_by_version.items():
        for response in responses:
            overall_question_n = response['overall_question_n']
            question_n = response['question_n']
            question_key = (overall_question_n, question_n)
            
            if question_key not in question_data:
                question_data[question_key] = {}
            
            question_data[question_key][version_key] = response
    
    # Process each unique question
    for question_key in sorted(question_data.keys()):
        overall_question_n, question_n = question_key
        version_responses = question_data[question_key]
        
        # Initialize row
        row = {
            'overall_question_n': overall_question_n,
            'question_n': question_n,
            'problem_sheet': ''
        }
        
        # Get problem sheet context from any available response
        for version_key in sorted(version_responses.keys()):
            response = version_responses[version_key]
            if 'question_details' in response and 'metadata' in response['question_details']:
                metadata = response['question_details']['metadata']
                if 'context' in metadata:
                    row['problem_sheet'] = metadata['context']
                    break
        
        # Add reasoning outputs for each version
        for version_key in sorted(responses_by_version.keys(), key=lambda x: int(x[1:])):
            if version_key in version_responses:
                reasoning_output = version_responses[version_key].get('reasoning_output', '')
                row[f'reasoning_output_{version_key}'] = reasoning_output
            else:
                row[f'reasoning_output_{version_key}'] = ''
        
        rows.append(row)
    
    return rows

def save_to_csv(rows, output_path):
    """
    Save the extracted reasoning data to CSV file.
    """
    if not rows:
        print("No data to save.")
        return
    
    # Get all column names in the desired order
    columns = ['overall_question_n', 'question_n', 'problem_sheet']
    
    # Add reasoning output columns (sorted by version number)
    reasoning_columns = [col for col in rows[0].keys() if col.startswith('reasoning_output_')]
    reasoning_columns.sort(key=lambda x: int(x.split('_v')[1]) if '_v' in x else int(x.split('_')[2][1:]))
    columns.extend(reasoning_columns)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Saved {len(rows)} rows to {output_path}")

def main():
    """
    Main function to process all chained prompt responses and create reasoning outputs CSV.
    """
    parser = argparse.ArgumentParser(description='Extract reasoning outputs from chained prompt responses')
    parser.add_argument('chained_prompts_dir', help='Path to the chained_prompts directory')
    parser.add_argument('sample_size', type=int, help='Number of response files to process')
    
    args = parser.parse_args()
    
    print(f"Loading chained prompt responses from {args.chained_prompts_dir}...")
    responses_by_version = load_chained_prompt_responses(args.chained_prompts_dir, args.sample_size)
    print(f"Loaded responses from {len(responses_by_version)} versions: {list(responses_by_version.keys())}")
    
    print("Extracting reasoning data...")
    rows = extract_question_reasoning_data(responses_by_version)
    
    print("Saving to CSV...")
    # Create openrouter_analysis directory if it doesn't exist
    analysis_dir = Path("openrouter_analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Generate systematic filename
    output_filename = f"reasoning_outputs_chained_prompt_{args.sample_size}.csv"
    output_path = analysis_dir / output_filename
    
    save_to_csv(rows, output_path)
    
    print(f"Done! Total questions: {len(rows)}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()