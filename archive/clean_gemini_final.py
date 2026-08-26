#!/usr/bin/env python3
"""
Script to clean the gemini_shuffle CSV by truncating IMPROPER PARSING fields.
"""

import csv
import sys
import os

def clean_gemini_csv():
    input_file = 'openrouter_analysis/subquestion_eval_gemini_shuffle_32.csv'
    output_file = 'openrouter_analysis/subquestion_eval_gemini_shuffle_32_cleaned.csv'
    
    # Increase CSV field size limit
    csv.field_size_limit(sys.maxsize)
    
    print(f"Cleaning {input_file}...")
    
    cleaned_fields = 0
    total_rows = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        header = next(reader)
        writer.writerow(header)
        
        for row in reader:
            total_rows += 1
            
            # Clean any field that starts with "IMPROPER PARSING"
            for i in range(len(row)):
                if row[i].startswith('IMPROPER PARSING'):
                    if row[i] != 'IMPROPER PARSING':  # Only clean if it has extra text
                        row[i] = 'IMPROPER PARSING'
                        cleaned_fields += 1
            
            writer.writerow(row)
            
            if total_rows % 100 == 0:
                print(f"Processed {total_rows} rows, cleaned {cleaned_fields} fields...")
    
    print(f"Completed! Processed {total_rows} rows, cleaned {cleaned_fields} IMPROPER PARSING fields.")
    
    # Check file sizes
    original_size = os.path.getsize(input_file) / (1024 * 1024)
    cleaned_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Original file size: {original_size:.1f} MB")
    print(f"Cleaned file size: {cleaned_size:.1f} MB")
    print(f"Size reduction: {original_size - cleaned_size:.1f} MB ({((original_size - cleaned_size) / original_size * 100):.1f}%)")

if __name__ == "__main__":
    clean_gemini_csv()