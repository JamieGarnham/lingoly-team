#!/usr/bin/env python3
"""
Create benchmark_single_obf_dev.jsonl from benchmark_same_obf.jsonl
This includes only the development/test questions (Question-dev split)
"""

import json

def create_dev_benchmark():
    """Create dev benchmark from same_obf dataset"""
    input_file = '../data/splits/benchmark_same_obf.jsonl'
    output_file = '../data/splits/benchmark_single_obf_dev.jsonl'
    
    dev_count = 0
    total_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line.strip())
            split_key = data['split_key']
            
            # Include only Question-dev entries (excluding Question-train)
            if 'Question-dev' in split_key:
                outfile.write(line)
                dev_count += 1
    
    print(f"Created benchmark_single_obf_dev.jsonl:")
    print(f"  Total entries processed: {total_count}")
    print(f"  Dev entries included: {dev_count}")
    print(f"  Output file: {output_file}")

if __name__ == "__main__":
    create_dev_benchmark()