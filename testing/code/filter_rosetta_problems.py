#!/usr/bin/env python3
"""
Filter benchmark_same_obf.jsonl to only include Rosetta problems
"""

import json
import csv

def get_rosetta_problem_numbers():
    """Get all Rosetta problem numbers from past-exam-papers.csv"""
    rosetta_problems = []
    with open('../data/past-exam-papers.csv', 'r') as f:
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
    
    input_file = '../data/splits/benchmark_same_obf.jsonl'
    output_file = '../data/splits/benchmark_same_obf_rosetta.jsonl'
    
    filtered_count = 0
    total_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            total_count += 1
            data = json.loads(line.strip())
            overall_question_n = data['index'][0]  # First element is overall_question_n
            
            if overall_question_n in rosetta_problems:
                outfile.write(line)
                filtered_count += 1
    
    print(f"Filtered {filtered_count} entries out of {total_count} total entries")
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    filter_benchmark()