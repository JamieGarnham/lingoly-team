#!/usr/bin/env python3
"""
Script to add format_fixed column to CSV files by joining with question_formats_fixed.csv
"""

import csv
import sys
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

def main():
    if len(sys.argv) != 3:
        print("Usage: python add_format_fixed.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Load format_fixed lookup
    format_file = "question_formats_fixed.csv"
    print(f"Loading format lookup from {format_file}")
    format_lookup = load_format_fixed_lookup(format_file)
    print(f"Loaded {len(format_lookup)} format mappings")
    
    # Add format_fixed column
    print(f"Processing {input_file} -> {output_file}")
    add_format_fixed_column(input_file, output_file, format_lookup)
    print("Done!")

if __name__ == "__main__":
    main()