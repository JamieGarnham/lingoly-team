import csv
import os

# Increase CSV field size limit to maximum to handle very large fields
import sys
csv.field_size_limit(sys.maxsize)

input_file = 'openrouter_analysis/subquestion_eval_gemini_shuffle_32.csv'
output_file = 'openrouter_analysis/subquestion_eval_gemini_shuffle_32_cleaned.csv'

print(f"Cleaning {input_file}...")

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Write header
    header = next(reader)
    writer.writerow(header)
    
    rows_processed = 0
    cleaned_answers = 0
    
    for row in reader:
        rows_processed += 1
        
        # Clean model answer columns (columns 6-37: model_answer_v1 to model_answer_v32)
        for i in range(6, 38):
            if len(row) > i and row[i].startswith('IMPROPER PARSING:'):
                row[i] = 'IMPROPER PARSING'
                cleaned_answers += 1
        
        writer.writerow(row)
        
        if rows_processed % 10000 == 0:
            print(f"Processed {rows_processed} rows, cleaned {cleaned_answers} answers...")

print(f"Completed! Processed {rows_processed} rows, cleaned {cleaned_answers} improper parsing answers.")

# Check file sizes
original_size = os.path.getsize(input_file) / (1024 * 1024)
cleaned_size = os.path.getsize(output_file) / (1024 * 1024)
print(f"Original file size: {original_size:.1f} MB")
print(f"Cleaned file size: {cleaned_size:.1f} MB")
print(f"Size reduction: {original_size - cleaned_size:.1f} MB ({((original_size - cleaned_size) / original_size * 100):.1f}%)")