#!/usr/bin/env python3
"""
Create final summary from JSONL data directly.
"""

import json
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


def main():
    original_path = 'testing_17/data/splits/benchmark_single_obf.jsonl'
    shuffled_path = 'testing_17/data/splits/benchmark_single_obf_shuffled.jsonl'
    
    # Load both datasets
    original_entries = []
    shuffled_entries = []
    
    with open(original_path, 'r', encoding='utf-8') as f:
        for line in f:
            original_entries.append(json.loads(line.strip()))
            
    with open(shuffled_path, 'r', encoding='utf-8') as f:
        for line in f:
            shuffled_entries.append(json.loads(line.strip()))
    
    # Analyze
    total_entries = len(original_entries)
    shuffled_entries_count = 0
    total_subprompts = 0
    shuffled_subprompts = 0
    
    shuffled_entry_indices = []
    
    for i, (orig_entry, shuf_entry) in enumerate(zip(original_entries, shuffled_entries)):
        was_entry_shuffled = was_shuffled(orig_entry, shuf_entry)
        
        if was_entry_shuffled:
            shuffled_entries_count += 1
            shuffled_entry_indices.append(i + 1)
        
        # Count subprompts
        subprompts = shuf_entry.get('question_details', {}).get('subprompts', [])
        if subprompts:
            for subprompt in subprompts:
                total_subprompts += 1
                if was_entry_shuffled:
                    shuffled_subprompts += 1
        else:
            total_subprompts += 1
            if was_entry_shuffled:
                shuffled_subprompts += 1
    
    print("=== FINAL SUMMARY ===")
    print(f"Original dataset: {original_path}")
    print(f"Shuffled dataset: {shuffled_path}")
    print()
    print("=== Entry-level Statistics ===")
    print(f"Total entries: {total_entries}")
    print(f"Entries with shuffled content: {shuffled_entries_count}")
    print(f"Entries without shuffling: {total_entries - shuffled_entries_count}")
    print(f"Percentage of entries shuffled: {shuffled_entries_count/total_entries*100:.1f}%")
    print()
    print("=== Subprompt-level Statistics ===")
    print(f"Total subprompts: {total_subprompts}")
    print(f"Subprompts from shuffled entries: {shuffled_subprompts}")
    print(f"Subprompts from non-shuffled entries: {total_subprompts - shuffled_subprompts}")
    print(f"Percentage of subprompts from shuffled entries: {shuffled_subprompts/total_subprompts*100:.1f}%")
    print()
    print("=== Files Created ===")
    print("1. shuffle_benchmark_tables.py - Main shuffling script")
    print("2. testing_17/data/splits/benchmark_single_obf_shuffled.jsonl - Shuffled dataset")
    print("3. testing_17/data/splits/benchmark_single_obf_clean.csv - CSV version with shuffle flag")
    print()
    print("The CSV includes these columns:")
    print("- entry_index: Sequential entry number")
    print("- question_n: Question identifier")
    print("- main_prompt: The main question prompt")
    print("- subprompt_*: Individual subquestion details")
    print("- context_preview: First 200 chars of the context")
    print("- num_table_rows: Number of detected table rows")
    print("- was_shuffled: Boolean flag indicating if entry was shuffled")
    print("- Other metadata fields")


if __name__ == '__main__':
    main()