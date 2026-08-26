#!/usr/bin/env python3
"""
Create benchmark_single_obf_dev.jsonl from benchmark_same_obf.jsonl
This includes only the development/test questions (Question-dev split)
"""

from zip_io import read_jsonl_zip, write_jsonl_zip

def create_dev_benchmark():
    """Create dev benchmark from same_obf dataset"""
    input_file = '../../data/splits/benchmark_same_obf.jsonl.zip'
    output_file = '../../data/splits/benchmark_single_obf_dev.jsonl.zip'

    dev_entries = []
    total_count = 0

    for data in read_jsonl_zip(input_file):
        total_count += 1
        split_key = data['split_key']

        # Include only Question-dev entries (excluding Question-train)
        if 'Question-dev' in split_key:
            dev_entries.append(data)

    write_jsonl_zip(output_file, dev_entries)

    print(f"Created benchmark_single_obf_dev.jsonl:")
    print(f"  Total entries processed: {total_count}")
    print(f"  Dev entries included: {len(dev_entries)}")
    print(f"  Output file: {output_file}")

if __name__ == "__main__":
    create_dev_benchmark()
