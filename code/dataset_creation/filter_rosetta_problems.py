#!/usr/bin/env python3
"""
Filter benchmark_same_obf.jsonl to only include Rosetta problems
"""

import csv

from zip_io import read_jsonl_zip, write_jsonl_zip

def get_rosetta_problem_numbers():
    """Get all Rosetta problem numbers from past-exam-papers.csv"""
    rosetta_problems = []
    with open('../../data/past-exam-papers.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) > 3 and 'Rosetta' in row[3]:
                rosetta_problems.append(int(row[0]))
    return set(rosetta_problems)

def filter_benchmark():
    """Filter benchmark_same_obf.jsonl to only Rosetta problems"""
    rosetta_problems = get_rosetta_problem_numbers()
    print(f"Found {len(rosetta_problems)} Rosetta problems")
    
    input_file = '../../data/splits/benchmark_same_obf.jsonl.zip'
    output_file = '../../data/splits/benchmark_same_obf_rosetta.jsonl.zip'

    filtered_entries = []
    total_count = 0

    for data in read_jsonl_zip(input_file):
        total_count += 1
        overall_question_n = data['index'][0]  # First element is overall_question_n

        if overall_question_n in rosetta_problems:
            filtered_entries.append(data)

    write_jsonl_zip(output_file, filtered_entries)

    print(f"Filtered {len(filtered_entries)} entries out of {total_count} total entries")
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    filter_benchmark()