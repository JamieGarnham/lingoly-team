#!/usr/bin/env python3
import json
import csv
import sys

def create_shuffle_jsonl():
    # Read the original JSONL file
    original_jsonl_path = "testing/data/splits/benchmark_same_obf.jsonl"
    shuffle_csv_path = "testing/data/splits/benchmark_same_obf_shuffle.csv"
    output_jsonl_path = "testing/data/splits/benchmark_same_obf_shuffle.jsonl"
    
    # Read CSV data into a lookup dictionary
    csv_contexts = {}
    with open(shuffle_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Debug: print first few rows to understand structure
            if len(csv_contexts) < 3:
                print(f"CSV row keys: {list(row.keys())}")
                print(f"CSV row sample: overall_question_n={row.get('overall_question_n')}, obfuscated_question_n={row.get('obfuscated_question_n')}, question_n={row.get('question_n')}")
            
            # The first column appears to be the overall_question_n, not the 'format' column
            overall_q_n = int(row['format'])  # This seems to be the actual overall_question_n
            obf_q_n = row['overall_question_n']  # This seems to be the actual obfuscated_question_n
            q_n = row['obfuscated']  # This seems to be the actual question_n
            
            # Wait, let me re-examine this - the column mapping seems wrong
            # Let me use the original approach but debug it
            try:
                key = (int(row['overall_question_n']), row['obfuscated_question_n'], row['question_n'])
                csv_contexts[key] = row['context']
            except (KeyError, ValueError) as e:
                print(f"Error processing row: {e}")
                print(f"Row data: {row}")
                break
    
    # Process JSONL file
    with open(original_jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            # Create lookup key from JSONL data
            overall_q_n = data['question_details']['metadata']['overall_question_n']
            obf_q_n = data['question_details']['metadata']['obfuscated_question_n']
            q_n = data['question_details']['question_n']
            key = (overall_q_n, obf_q_n, q_n)
            
            # Replace context if found in CSV
            if key in csv_contexts:
                data['question_details']['metadata']['context'] = csv_contexts[key]
            else:
                print(f"Warning: No shuffled context found for {key}")
            
            # Write modified JSON line
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Created {output_jsonl_path} with shuffled contexts")

if __name__ == "__main__":
    create_shuffle_jsonl()