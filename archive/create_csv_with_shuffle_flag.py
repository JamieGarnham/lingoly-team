#!/usr/bin/env python3
"""
Convert the shuffled JSONL dataset to CSV format with shuffle flag.
"""

import json
import csv
import re
from pathlib import Path


def detect_table_rows(text: str) -> list:
    """Detect potential table rows in text (same logic as shuffle script)."""
    lines = text.strip().split('\n')
    table_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip if line looks like a header or description
        if any(keyword in line.lower() for keyword in [
            'english', 'dialect', 'language x', 'pronunciation', 'note:', 'given below'
        ]):
            continue
            
        # Skip numbered story lines (these are ordered sequences)
        if re.match(r'^\d+\s+', line):
            continue
            
        # Skip lines that look like headers or section breaks
        if line.isupper() or '---' in line or '===' in line:
            continue
            
        # Look for tab-separated or multi-space separated content
        if '\t' in line or re.search(r'\s{2,}', line):
            parts = re.split(r'\t|\s{2,}', line)
            if len(parts) >= 2:
                if not any(desc_word in line.lower() for desc_word in [
                    'the following', 'below are', 'study the', 'answer by'
                ]):
                    table_rows.append(line)
    
    return table_rows


def is_order_sensitive(rows: list) -> bool:
    """Determine if rows contain order-sensitive content (same logic as shuffle script)."""
    if not rows:
        return False
        
    # Check for numbered items
    numbered_count = sum(1 for row in rows if re.match(r'^\d+\.?\s+', row.strip()))
    if numbered_count >= 2:
        return True
        
    # Check for lettered items  
    lettered_count = sum(1 for row in rows if re.match(r'^[a-zA-Z]\.?\s+', row.strip()))
    if lettered_count >= 2:
        return True
        
    # Check for ordinal indicators
    ordinal_words = ['first', 'second', 'third', 'fourth', 'fifth', 'next', 'then', 'finally']
    content = ' '.join(rows).lower()
    if any(word in content for word in ordinal_words):
        return True
        
    # Check for match-up indicators
    match_indicators = ['match', 'correspond', 'pair', 'connect', 'link']
    if any(indicator in content for indicator in match_indicators):
        return True
        
    # Check for sequence indicators
    sequence_indicators = ['in order', 'sequence', 'step', 'stage', 'phase']
    if any(indicator in content for indicator in sequence_indicators):
        return True
        
    return False


def was_shuffled(original_entry, shuffled_entry):
    """
    Determine if an entry was actually shuffled by comparing original and shuffled contexts.
    """
    orig_context = original_entry.get('question_details', {}).get('metadata', {}).get('context', '')
    shuf_context = shuffled_entry.get('question_details', {}).get('metadata', {}).get('context', '')
    
    if orig_context == shuf_context:
        return False
        
    # Check if it has table rows that could be shuffled
    table_rows = detect_table_rows(orig_context)
    if len(table_rows) < 2:
        return False
        
    # Check if it's order-sensitive
    if is_order_sensitive(table_rows):
        return False
        
    return True


def convert_to_csv(original_jsonl_path, shuffled_jsonl_path, output_csv_path):
    """Convert shuffled JSONL to CSV with shuffle flag."""
    
    # Load both datasets
    original_entries = []
    shuffled_entries = []
    
    with open(original_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            original_entries.append(json.loads(line.strip()))
            
    with open(shuffled_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            shuffled_entries.append(json.loads(line.strip()))
    
    # Create CSV data
    csv_data = []
    
    for i, (orig_entry, shuf_entry) in enumerate(zip(original_entries, shuffled_entries)):
        # Extract key information
        question_details = shuf_entry.get('question_details', {})
        metadata = question_details.get('metadata', {})
        
        # Determine if this entry was shuffled
        shuffled_flag = was_shuffled(orig_entry, shuf_entry)
        
        # Extract subprompts and answers
        subprompts = question_details.get('subprompts', [])
        
        # Create a row for each subprompt
        if subprompts:
            for subprompt in subprompts:
                row = {
                    'entry_index': i + 1,
                    'overall_question_n': shuf_entry.get('index', [None, None, None, None, None])[0],
                    'obfuscated_question_n': shuf_entry.get('index', [None, None, None, None, None])[1],
                    'obfuscated': shuf_entry.get('index', [None, None, None, None, None])[2],
                    'obf_num': shuf_entry.get('index', [None, None, None, None, None])[3],
                    'question_n': question_details.get('question_n', ''),
                    'main_prompt': question_details.get('prompt', ''),
                    'subprompt_n': subprompt.get('questionpart_n', ''),
                    'subprompt_question': subprompt.get('question', ''),
                    'subprompt_answer': subprompt.get('answer', ''),
                    'preamble': metadata.get('preamble', ''),
                    'context': metadata.get('context', ''),
                    'was_shuffled': shuffled_flag,
                    'split_key': shuf_entry.get('split_key', '')
                }
                csv_data.append(row)
        else:
            # If no subprompts, create a single row
            row = {
                'entry_index': i + 1,
                'overall_question_n': shuf_entry.get('index', [None, None, None, None, None])[0],
                'obfuscated_question_n': shuf_entry.get('index', [None, None, None, None, None])[1],
                'obfuscated': shuf_entry.get('index', [None, None, None, None, None])[2],
                'obf_num': shuf_entry.get('index', [None, None, None, None, None])[3],
                'question_n': question_details.get('question_n', ''),
                'main_prompt': question_details.get('prompt', ''),
                'subprompt_n': '',
                'subprompt_question': '',
                'subprompt_answer': '',
                'preamble': metadata.get('preamble', ''),
                'context': metadata.get('context', ''),
                'was_shuffled': shuffled_flag,
                'split_key': shuf_entry.get('split_key', '')
            }
            csv_data.append(row)
    
    # Write CSV
    fieldnames = [
        'entry_index', 'overall_question_n', 'obfuscated_question_n', 'obfuscated', 
        'obf_num', 'question_n', 'main_prompt', 'subprompt_n', 'subprompt_question', 
        'subprompt_answer', 'preamble', 'context', 'was_shuffled', 'split_key'
    ]
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    return len(csv_data), sum(1 for row in csv_data if row['was_shuffled'])


def main():
    original_path = 'testing_17/data/splits/benchmark_single_obf.jsonl'
    shuffled_path = 'testing_17/data/splits/benchmark_single_obf_shuffled.jsonl'
    output_path = 'testing_17/data/splits/benchmark_single_obf_shuffled.csv'
    
    print(f"Converting shuffled dataset to CSV...")
    print(f"Original: {original_path}")
    print(f"Shuffled: {shuffled_path}")
    print(f"Output: {output_path}")
    
    total_rows, shuffled_rows = convert_to_csv(original_path, shuffled_path, output_path)
    
    print(f"\nCSV created successfully!")
    print(f"Total rows: {total_rows}")
    print(f"Shuffled entries: {shuffled_rows}")
    print(f"Non-shuffled entries: {total_rows - shuffled_rows}")
    print(f"Percentage shuffled: {shuffled_rows/total_rows*100:.1f}%")


if __name__ == '__main__':
    main()