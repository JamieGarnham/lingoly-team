#!/usr/bin/env python3
"""
Quick validation script to show the differences between original and shuffled datasets.
"""

import json

def compare_datasets(original_path, shuffled_path, num_examples=3):
    """Compare the first few entries to show shuffling differences."""
    
    with open(original_path, 'r') as f1, open(shuffled_path, 'r') as f2:
        for i in range(num_examples):
            orig_line = f1.readline().strip()
            shuf_line = f2.readline().strip()
            
            if not orig_line or not shuf_line:
                break
                
            orig_entry = json.loads(orig_line)
            shuf_entry = json.loads(shuf_line)
            
            orig_context = orig_entry.get('question_details', {}).get('metadata', {}).get('context', '')
            shuf_context = shuf_entry.get('question_details', {}).get('metadata', {}).get('context', '')
            
            if orig_context != shuf_context:
                print(f"=== Entry {i+1}: {orig_entry.get('question_details', {}).get('question_n', 'Unknown')} ===")
                print(f"Original table rows:")
                orig_lines = [line.strip() for line in orig_context.split('\n') if line.strip() and '\t' in line][:5]
                for j, line in enumerate(orig_lines, 1):
                    print(f"  {j}. {line}")
                
                print(f"\nShuffled table rows:")
                shuf_lines = [line.strip() for line in shuf_context.split('\n') if line.strip() and '\t' in line][:5]
                for j, line in enumerate(shuf_lines, 1):
                    print(f"  {j}. {line}")
                print()

if __name__ == '__main__':
    compare_datasets(
        'testing_17/data/splits/benchmark_single_obf.jsonl',
        'testing_17/data/splits/benchmark_single_obf_shuffled.jsonl',
        3
    )