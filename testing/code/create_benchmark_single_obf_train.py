#!/usr/bin/env python3
"""
Create benchmark_single_obf_train.jsonl from benchmark_same_obf.jsonl
This includes only the training questions (Question-train split)
"""

import json

def create_train_benchmark():
    """Create train benchmark from same_obf dataset"""
    input_file = '../data/splits/benchmark_same_obf.jsonl'
    output_file = '../data/splits/benchmark_single_obf_train.jsonl'
    
    train_count = 0
    total_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line.strip())
            split_key = data['split_key']
            
            # Include only Question-train entries (excluding Question-dev)
            if 'Question-train' in split_key:
                outfile.write(line)
                train_count += 1
    
    print(f"Created benchmark_single_obf_train.jsonl:")
    print(f"  Total entries processed: {total_count}")
    print(f"  Train entries included: {train_count}")
    print(f"  Output file: {output_file}")

if __name__ == "__main__":
    create_train_benchmark()