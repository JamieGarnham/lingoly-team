#!/usr/bin/env python3
"""
Script to shuffle tabular data in benchmark_single_obf dataset while preserving order-sensitive content.

This script creates a copy of the dataset with randomized table rows to mitigate LLMs' 
sensitivity to information presentation order, while carefully preserving the order 
of numbered/lettered examples and match-up problems.
"""

import json
import re
import random
import argparse
from typing import List, Dict, Any, Tuple
from pathlib import Path


def detect_table_rows(text: str) -> List[str]:
    """
    Detect potential table rows in text based on common patterns.
    
    Returns list of lines that appear to be table rows.
    """
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
        # that appears to be word-translation pairs
        if '\t' in line or re.search(r'\s{2,}', line):
            # Check if it looks like a word-translation pair
            parts = re.split(r'\t|\s{2,}', line)
            if len(parts) >= 2:
                # Further check: avoid lines that are clearly descriptions
                if not any(desc_word in line.lower() for desc_word in [
                    'the following', 'below are', 'study the', 'answer by'
                ]):
                    table_rows.append(line)
    
    return table_rows


def is_order_sensitive(rows: List[str]) -> bool:
    """
    Determine if the rows contain order-sensitive content that shouldn't be shuffled.
    
    Returns True if the order appears to be important (numbered, lettered, or matched content).
    """
    if not rows:
        return False
        
    # Check for numbered items
    numbered_count = sum(1 for row in rows if re.match(r'^\d+\.?\s+', row.strip()))
    if numbered_count >= 2:  # At least 2 numbered items suggests ordering matters
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


def shuffle_table_content(context: str) -> str:
    """
    Shuffle table rows in the context while preserving order-sensitive content.
    
    Returns the context with shuffled table rows.
    """
    # Detect potential table rows
    table_rows = detect_table_rows(context)
    
    if len(table_rows) < 2:
        # Not enough rows to shuffle
        return context
        
    # Check if content is order-sensitive
    if is_order_sensitive(table_rows):
        # Don't shuffle order-sensitive content
        return context
        
    # Create a shuffled copy of the rows
    shuffled_rows = table_rows.copy()
    random.shuffle(shuffled_rows)
    
    # Replace the original rows with shuffled ones
    result_context = context
    for original, shuffled in zip(table_rows, shuffled_rows):
        # Only replace if they're different to avoid unnecessary changes
        if original != shuffled:
            result_context = result_context.replace(original, shuffled, 1)
    
    return result_context


def process_dataset_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single dataset entry and shuffle its tabular content.
    
    Returns the modified entry.
    """
    # Create a deep copy of the entry to avoid modifying the original
    modified_entry = json.loads(json.dumps(entry))
    
    # Get the context field which contains the tabular data
    if 'question_details' in entry and 'metadata' in entry['question_details']:
        context = entry['question_details']['metadata'].get('context', '')
        
        if context:
            # Shuffle the table content
            shuffled_context = shuffle_table_content(context)
            modified_entry['question_details']['metadata']['context'] = shuffled_context
    
    return modified_entry


def main():
    parser = argparse.ArgumentParser(description='Shuffle tabular data in benchmark dataset')
    parser.add_argument('input_file', help='Input JSONL file path')
    parser.add_argument('output_file', help='Output JSONL file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without writing output')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return 1
        
    if not args.dry_run and output_path.exists():
        response = input(f"Output file {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted")
            return 1
    
    print(f"Processing {input_path}...")
    
    entries_processed = 0
    entries_modified = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        if args.dry_run:
            print("DRY RUN MODE - showing changes that would be made:")
            print()
            
        processed_entries = []
        
        for line_num, line in enumerate(infile, 1):
            try:
                entry = json.loads(line.strip())
                original_context = entry.get('question_details', {}).get('metadata', {}).get('context', '')
                
                modified_entry = process_dataset_entry(entry)
                modified_context = modified_entry.get('question_details', {}).get('metadata', {}).get('context', '')
                
                entries_processed += 1
                
                if original_context != modified_context:
                    entries_modified += 1
                    
                    if args.dry_run:
                        print(f"Entry {line_num} would be modified:")
                        print(f"  Question: {entry.get('question_details', {}).get('question_n', 'Unknown')}")
                        
                        # Show table rows that would be shuffled
                        orig_rows = detect_table_rows(original_context)
                        mod_rows = detect_table_rows(modified_context)
                        
                        if orig_rows != mod_rows:
                            print(f"  Table rows would be shuffled ({len(orig_rows)} rows)")
                            print(f"  Order-sensitive: {is_order_sensitive(orig_rows)}")
                        print()
                
                processed_entries.append(modified_entry)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                return 1
                
    print(f"Processed {entries_processed} entries")
    print(f"Modified {entries_modified} entries")
    
    if not args.dry_run:
        print(f"Writing to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for entry in processed_entries:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write('\n')
        print("Done!")
    else:
        print("Dry run completed. Use without --dry-run to actually create the shuffled dataset.")
    
    return 0


if __name__ == '__main__':
    exit(main())