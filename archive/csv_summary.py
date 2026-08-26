#!/usr/bin/env python3
"""Simple CSV summary without pandas dependency."""

import csv

# Increase CSV field size limit to handle large context fields
csv.field_size_limit(1000000)

def summarize_csv(csv_path):
    """Create summary statistics for the CSV file."""
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        total_rows = 0
        shuffled_count = 0
        shuffled_entries = set()
        non_shuffled_entries = set()
        
        for row in reader:
            total_rows += 1
            entry_idx = row['entry_index']
            
            if row['was_shuffled'].lower() == 'true':
                shuffled_count += 1
                shuffled_entries.add(entry_idx)
            else:
                non_shuffled_entries.add(entry_idx)
    
    print("=== CSV Summary ===")
    print(f"Total subprompts: {total_rows}")
    print(f"Shuffled subprompts: {shuffled_count}")
    print(f"Non-shuffled subprompts: {total_rows - shuffled_count}")
    print(f"Percentage shuffled: {shuffled_count/total_rows*100:.1f}%")
    print()
    print(f"Unique entries with shuffled content: {len(shuffled_entries)}")
    print(f"Unique entries without shuffling: {len(non_shuffled_entries)}")
    
    # Show a few examples
    print("\n=== Sample Shuffled Questions ===")
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            if row['was_shuffled'].lower() == 'true' and count < 5:
                print(f"Entry {row['entry_index']}: {row['question_n']} - {row['main_prompt'][:80]}...")
                count += 1

if __name__ == '__main__':
    summarize_csv('testing_17/data/splits/benchmark_single_obf_shuffled.csv')