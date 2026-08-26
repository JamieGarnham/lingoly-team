#!/usr/bin/env python3
"""
Create benchmark_same_obf.jsonl with consistent obfuscation per problem sheet (same overall_question_n)
and add format column from subquestion_eval CSV.
"""

import json
import csv
from collections import defaultdict
import ast

def load_format_mapping():
    """Load format information from subquestion_eval CSV"""
    format_mapping = {}
    
    csv.field_size_limit(1000000)
    with open('../openrouter_analysis/original_prompt/subquestion_eval_original_prompt_32.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            overall_q_n = int(row['overall_question_n'])
            format_value = row['format']
            
            # Parse format value if it's a string representation of a list
            if format_value.startswith('[') and format_value.endswith(']'):
                try:
                    format_list = ast.literal_eval(format_value)
                    format_value = format_list[0] if format_list else "Unknown"
                except:
                    format_value = "Unknown"
            
            format_mapping[overall_q_n] = format_value
    
    print(f"Loaded format mapping for {len(format_mapping)} overall questions")
    return format_mapping

def create_benchmark_same_obf():
    """Create new benchmark with consistent obfuscation per problem sheet"""
    
    # Load format mapping
    format_mapping = load_format_mapping()
    
    # First pass: collect all entries and group by overall_question_n
    entries_by_overall_q = defaultdict(list)
    
    with open('data/splits/benchmark_single_obf.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                overall_q_n = entry['question_details']['metadata']['overall_question_n']
                entries_by_overall_q[overall_q_n].append(entry)
    
    print(f"Found entries for {len(entries_by_overall_q)} overall questions")
    
    # Second pass: for each overall_question_n, choose one obfuscation and apply to all
    output_entries = []
    
    for overall_q_n, entries in entries_by_overall_q.items():
        # Choose the first obfuscation for this overall question number
        chosen_obf_entry = entries[0]
        chosen_obf_num = chosen_obf_entry['question_details']['metadata']['obf_num']
        chosen_mapping = chosen_obf_entry['question_details']['metadata']['mapping']
        chosen_obf_question_n = chosen_obf_entry['question_details']['metadata']['obfuscated_question_n']
        
        print(f"Overall Q{overall_q_n}: Using obfuscation {chosen_obf_num} for {len(entries)} questions")
        
        # Apply this obfuscation to all entries in this overall question
        for entry in entries:
            # Create new entry with consistent obfuscation
            new_entry = entry.copy()
            
            # Update the index
            new_entry['index'][1] = chosen_obf_question_n  # obfuscated_question_n
            new_entry['index'][3] = chosen_obf_num  # obf_num
            
            # Update metadata
            new_entry['question_details']['metadata']['obf_num'] = chosen_obf_num
            new_entry['question_details']['metadata']['mapping'] = chosen_mapping
            new_entry['question_details']['metadata']['obfuscated_question_n'] = chosen_obf_question_n
            
            # Ensure obfuscated is True
            new_entry['question_details']['metadata']['obfuscated'] = "True"
            new_entry['index'][2] = "True"
            
            # Add format as first element in index
            format_value = format_mapping.get(overall_q_n, "Unknown")
            new_entry['index'] = [format_value] + new_entry['index']
            
            output_entries.append(new_entry)
    
    # Sort by overall_question_n, then by question_n for consistency
    output_entries.sort(key=lambda x: (x['question_details']['metadata']['overall_question_n'], 
                                     x['question_details']['question_n']))
    
    # Write output
    output_file = 'data/splits/benchmark_same_obf.jsonl'
    with open(output_file, 'w') as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\\nCreated {output_file} with {len(output_entries)} entries")
    
    # Summary statistics
    obf_usage = defaultdict(int)
    format_usage = defaultdict(int)
    
    for entry in output_entries:
        obf_num = entry['question_details']['metadata']['obf_num']
        format_val = entry['index'][0]
        obf_usage[obf_num] += 1
        format_usage[format_val] += 1
    
    print(f"\\nObfuscation usage:")
    for obf_num, count in sorted(obf_usage.items()):
        print(f"  Obfuscation {obf_num}: {count} entries")
    
    print(f"\\nFormat distribution:")
    for format_val, count in sorted(format_usage.items()):
        print(f"  {format_val}: {count} entries")

if __name__ == "__main__":
    create_benchmark_same_obf()