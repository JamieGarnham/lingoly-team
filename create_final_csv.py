#!/usr/bin/env python3
"""
Create final CSV with 173 entries + header, full context text with proper newline handling.
"""

import json
import csv
import re


def detect_table_rows(text: str) -> list:
    """Detect potential table rows in text."""
    lines = text.strip().split('\n')
    table_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if any(keyword in line.lower() for keyword in [
            'english', 'dialect', 'language x', 'pronunciation', 'note:', 'given below'
        ]):
            continue
            
        if re.match(r'^\d+\s+', line):
            continue
            
        if line.isupper() or '---' in line or '===' in line:
            continue
            
        if '\t' in line or re.search(r'\s{2,}', line):
            parts = re.split(r'\t|\s{2,}', line)
            if len(parts) >= 2:
                if not any(desc_word in line.lower() for desc_word in [
                    'the following', 'below are', 'study the', 'answer by'
                ]):
                    table_rows.append(line)
    
    return table_rows


def is_order_sensitive(rows: list) -> bool:
    """Determine if rows contain order-sensitive content."""
    if not rows:
        return False
        
    numbered_count = sum(1 for row in rows if re.match(r'^\d+\.?\s+', row.strip()))
    if numbered_count >= 2:
        return True
        
    lettered_count = sum(1 for row in rows if re.match(r'^[a-zA-Z]\.?\s+', row.strip()))
    if lettered_count >= 2:
        return True
        
    ordinal_words = ['first', 'second', 'third', 'fourth', 'fifth', 'next', 'then', 'finally']
    content = ' '.join(rows).lower()
    if any(word in content for word in ordinal_words):
        return True
        
    match_indicators = ['match', 'correspond', 'pair', 'connect', 'link']
    if any(indicator in content for indicator in match_indicators):
        return True
        
    sequence_indicators = ['in order', 'sequence', 'step', 'stage', 'phase']
    if any(indicator in content for indicator in sequence_indicators):
        return True
        
    return False


def was_shuffled(original_entry, shuffled_entry):
    """Determine if an entry was actually shuffled."""
    orig_context = original_entry.get('question_details', {}).get('metadata', {}).get('context', '')
    shuf_context = shuffled_entry.get('question_details', {}).get('metadata', {}).get('context', '')
    
    if orig_context == shuf_context:
        return False
        
    table_rows = detect_table_rows(orig_context)
    if len(table_rows) < 2:
        return False
        
    if is_order_sensitive(table_rows):
        return False
        
    return True


def create_final_csv(original_jsonl_path, shuffled_jsonl_path, output_csv_path):
    """Create CSV with full context and proper formatting."""
    
    # Load both datasets
    original_entries = []
    shuffled_entries = []
    
    with open(original_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            original_entries.append(json.loads(line.strip()))
            
    with open(shuffled_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            shuffled_entries.append(json.loads(line.strip()))
    
    csv_data = []
    
    for i, (orig_entry, shuf_entry) in enumerate(zip(original_entries, shuffled_entries)):
        question_details = shuf_entry.get('question_details', {})
        metadata = question_details.get('metadata', {})
        
        # Determine if this entry was shuffled
        shuffled_flag = was_shuffled(orig_entry, shuf_entry)
        
        # Get table information
        context = metadata.get('context', '')
        table_rows = detect_table_rows(context)
        num_table_rows = len(table_rows)
        
        # Count subprompts
        subprompts = question_details.get('subprompts', [])
        num_subprompts = len(subprompts)
        
        # Get first subprompt info only (avoid JSON with commas)
        first_subprompt = subprompts[0] if subprompts else {}
        first_subprompt_q = first_subprompt.get('questionpart_n', '')
        first_subprompt_text = first_subprompt.get('question', '').replace('\n', '\\n').replace('\r', '\\r')
        first_subprompt_answer = str(first_subprompt.get('answer', '')).replace('\n', '\\n').replace('\r', '\\r')
        
        # Replace newlines with escaped newlines for CSV compatibility
        escaped_context = context.replace('\n', '\\n').replace('\r', '\\r')
        escaped_preamble = metadata.get('preamble', '').replace('\n', '\\n').replace('\r', '\\r')
        escaped_prompt = question_details.get('prompt', '').replace('\n', '\\n').replace('\r', '\\r')
        
        row = {
            'entry_index': i + 1,
            'overall_question_n': shuf_entry.get('index', [None])[0],
            'obfuscated_question_n': shuf_entry.get('index', [None, None])[1] if len(shuf_entry.get('index', [])) > 1 else None,
            'obfuscated': shuf_entry.get('index', [None, None, None])[2] if len(shuf_entry.get('index', [])) > 2 else None,
            'obf_num': shuf_entry.get('index', [None, None, None, None])[3] if len(shuf_entry.get('index', [])) > 3 else None,
            'question_n': question_details.get('question_n', ''),
            'main_prompt': escaped_prompt,
            'num_subprompts': num_subprompts,
            'first_subprompt_n': first_subprompt_q,
            'first_subprompt_question': first_subprompt_text,
            'first_subprompt_answer': first_subprompt_answer,
            'preamble': escaped_preamble,
            'context': escaped_context,  # Escaped newlines for CSV
            'num_table_rows': num_table_rows,
            'was_shuffled': shuffled_flag,
            'split_key': shuf_entry.get('split_key', '')
        }
        csv_data.append(row)
    
    # Write CSV with proper quoting
    fieldnames = [
        'entry_index', 'overall_question_n', 'obfuscated_question_n', 'obfuscated', 
        'obf_num', 'question_n', 'main_prompt', 'num_subprompts', 
        'first_subprompt_n', 'first_subprompt_question', 'first_subprompt_answer',
        'preamble', 'context', 'num_table_rows', 'was_shuffled', 'split_key'
    ]
    
    # Write CSV with explicit newline handling
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, 
                               quoting=csv.QUOTE_ALL, 
                               lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_data)
    
    return len(csv_data), sum(1 for row in csv_data if row['was_shuffled'])


def main():
    original_path = 'testing_17/data/splits/benchmark_single_obf.jsonl'
    shuffled_path = 'testing_17/data/splits/benchmark_single_obf_shuffled.jsonl'
    output_path = 'testing_17/data/splits/benchmark_single_obf_final.csv'
    
    print(f"Creating final CSV with full context...")
    total_entries, shuffled_entries = create_final_csv(original_path, shuffled_path, output_path)
    
    print(f"\nFinal CSV created: {output_path}")
    print(f"Total entries: {total_entries}")
    print(f"Shuffled entries: {shuffled_entries}")
    print(f"Non-shuffled entries: {total_entries - shuffled_entries}")
    print(f"Percentage shuffled: {shuffled_entries/total_entries*100:.1f}%")
    print(f"Total rows in CSV: {total_entries + 1} (including header)")
    
    # Verify row count
    with open(output_path, 'r', encoding='utf-8') as f:
        actual_rows = sum(1 for line in f)
    print(f"Verified: CSV has {actual_rows} total rows")


if __name__ == '__main__':
    main()