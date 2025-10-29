#!/usr/bin/env python3
"""
Script to process all CSV files in openrouter_analysis directory:
1. Add format_fixed column by joining with question_formats_fixed.csv
2. Replace 'combined_shuffle_original' with 'deepseek_combined_shuffle' in names
3. Add 'fix' to the end of filenames
4. Create new files without modifying originals
"""

import csv
import sys
import os
from pathlib import Path

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

def load_format_fixed_lookup(format_file_path):
    """Load the format_fixed data into a lookup dictionary"""
    lookup = {}
    with open(format_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['overall_question_n'], row['question_n'])
            lookup[key] = row['format_fixed']
    return lookup

def add_format_fixed_column(input_file, output_file, format_lookup):
    """Add format_fixed column to the CSV file"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        
        # Add format_fixed to fieldnames
        fieldnames = reader.fieldnames + ['format_fixed']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            # Create lookup key
            key = (row['overall_question_n'], row['question_n'])
            
            # Add format_fixed value
            row['format_fixed'] = format_lookup.get(key, 'Unknown')
            
            writer.writerow(row)

def process_file(input_path, format_lookup, base_dir):
    """Process a single CSV file"""
    input_file = Path(input_path)
    filename = input_file.stem  # filename without extension
    extension = input_file.suffix
    
    # Replace 'combined_shuffle_original' with 'deepseek_combined_shuffle' in filename
    new_filename = filename.replace('combined_shuffle_original', 'deepseek_combined_shuffle')
    
    # Add 'fix' to the end
    new_filename = new_filename + '_fix' + extension
    
    # Create output path
    output_path = base_dir / new_filename
    
    print(f"Processing: {input_file.name} -> {new_filename}")
    add_format_fixed_column(input_path, output_path, format_lookup)
    return output_path

def main():
    # Setup paths
    base_dir = Path("openrouter_analysis")
    format_file = "question_formats_fixed.csv"
    
    # Load format_fixed lookup
    print(f"Loading format lookup from {format_file}")
    format_lookup = load_format_fixed_lookup(format_file)
    print(f"Loaded {len(format_lookup)} format mappings")
    
    # Find all CSV files directly in the openrouter_analysis directory (not subdirectories)
    csv_files = []
    for file in os.listdir(base_dir):
        file_path = base_dir / file
        if file.endswith('.csv') and file_path.is_file() and file.startswith('subquestion_eval_'):
            # Skip files that already have 'fix' in the name
            if '_fix' not in file:
                csv_files.append(file_path)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    processed_files = []
    for csv_file in csv_files:
        try:
            output_path = process_file(csv_file, format_lookup, base_dir)
            processed_files.append(output_path)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    print(f"\nProcessed {len(processed_files)} files successfully:")
    for file_path in processed_files:
        print(f"  - {file_path.name}")

if __name__ == "__main__":
    main()