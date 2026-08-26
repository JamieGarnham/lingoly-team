#!/usr/bin/env python3
"""
Create benchmark_same_obf.jsonl by selecting one random obfuscation per problem sheet
from benchmark.jsonl and adding format information.
"""

import json
import csv
from collections import defaultdict
import random
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
    """Create new benchmark with one randomly selected obfuscation per problem sheet"""
    
    # Load format mapping
    format_mapping = load_format_mapping()
    
    # First pass: group all entries by overall_question_n and obfuscated_question_n
    # Only include entries where obfuscated is True
    entries_by_overall_and_obf = defaultdict(lambda: defaultdict(list))
    
    with open('data/splits/benchmark.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Only process obfuscated entries
                if entry['question_details']['metadata']['obfuscated'] == "True":
                    overall_q_n = entry['question_details']['metadata']['overall_question_n']
                    obfuscated_q_n = entry['question_details']['metadata']['obfuscated_question_n']
                    entries_by_overall_and_obf[overall_q_n][obfuscated_q_n].append(entry)
    
    print(f"Found obfuscated entries for {len(entries_by_overall_and_obf)} overall questions")
    
    # Second pass: for each overall_question_n, randomly select one obfuscation
    output_entries = []
    random.seed(42)  # For reproducible results
    
    for overall_q_n, obfuscations in entries_by_overall_and_obf.items():
        # Randomly select one obfuscation for this overall question
        available_obfuscations = list(obfuscations.keys())
        chosen_obfuscation = random.choice(available_obfuscations)
        
        # Get all entries for this chosen obfuscation
        chosen_entries = obfuscations[chosen_obfuscation]
        
        print(f"Overall Q{overall_q_n}: Selected obfuscation {chosen_obfuscation} ({len(chosen_entries)} questions) from {len(available_obfuscations)} available")
        
        # Add format information to each entry and add to output
        for entry in chosen_entries:
            # Create new entry with format added to index
            new_entry = entry.copy()
            
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
    
    print(f"\nCreated {output_file} with {len(output_entries)} entries")
    
    # Summary statistics
    obf_usage = defaultdict(int)
    format_usage = defaultdict(int)
    overall_q_usage = defaultdict(int)
    
    for entry in output_entries:
        obfuscated_q_n = entry['question_details']['metadata']['obfuscated_question_n']
        format_val = entry['index'][0]
        overall_q_n = entry['question_details']['metadata']['overall_question_n']
        
        obf_usage[obfuscated_q_n] += 1
        format_usage[format_val] += 1
        overall_q_usage[overall_q_n] += 1
    
    print(f"\nSelected obfuscations (showing first 10):")
    for obf_q_n, count in sorted(list(obf_usage.items())[:10]):
        print(f"  {obf_q_n}: {count} entries")
    
    print(f"\nFormat distribution:")
    for format_val, count in sorted(format_usage.items()):
        print(f"  {format_val}: {count} entries")
    
    print(f"\nOverall questions represented: {len(overall_q_usage)}")

if __name__ == "__main__":
    create_benchmark_same_obf()