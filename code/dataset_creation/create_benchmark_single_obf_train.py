#!/usr/bin/env python3
"""
Create benchmark_single_obf_train.jsonl from benchmark_same_obf.jsonl
This includes only the training questions (Question-train split)
"""

from zip_io import read_jsonl_zip, write_jsonl_zip

def create_train_benchmark():
    """Create train benchmark from same_obf dataset"""
    input_file = '../../data/splits/benchmark_same_obf.jsonl.zip'
    output_file = '../../data/splits/benchmark_single_obf_train.jsonl.zip'

    train_entries = []
    total_count = 0

    for data in read_jsonl_zip(input_file):
        total_count += 1
        split_key = data['split_key']

        # Include only Question-train entries (excluding Question-dev)
        if 'Question-train' in split_key:
            train_entries.append(data)

    write_jsonl_zip(output_file, train_entries)

    print(f"Created benchmark_single_obf_train.jsonl:")
    print(f"  Total entries processed: {total_count}")
    print(f"  Train entries included: {len(train_entries)}")
    print(f"  Output file: {output_file}")

if __name__ == "__main__":
    create_train_benchmark()
