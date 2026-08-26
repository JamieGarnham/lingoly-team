#!/usr/bin/env python3
"""
Script to combine judge evaluation files across all problem formats.
Creates organized combined files by model and sample size.
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict

def parse_filename(filename):
    """Parse filename to extract model, size, format, and judge type"""
    # Example: deepseek-r1_deepseek-combined-shuffle-32-fix_rosetta_judge_evaluation_20251014_214625.jsonl
    parts = filename.split('_')
    
    # Extract model name (first part before first underscore)
    model_part = parts[0]
    
    # Extract size (look for pattern like "32-fix" or "16-fix")
    size = None
    format_name = None
    judge_type = None
    
    for i, part in enumerate(parts):
        if '-fix' in part:
            # Extract size from this part (e.g., "32-fix" -> "32")
            size = part.split('-fix')[0].split('-')[-1]
        elif part in ['rosetta', 'pattern', 'monolingual', 'match-up', 'unknown']:
            format_name = part
        elif part in ['judge', 'mc']:
            if part == 'mc':
                judge_type = 'mc_judge'
            else:
                judge_type = 'judge'
            break
    
    # Determine model family
    if 'deepseek' in model_part:
        model_family = 'deepseek'
    elif 'gemini' in model_part:
        model_family = 'gemini'
    elif 'llama' in model_part:
        model_family = 'llama'
    else:
        model_family = 'unknown'
    
    return model_family, size, format_name, judge_type

def load_jsonl_file(filepath):
    """Load JSONL file and return list of JSON objects"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl_file(data, filepath):
    """Save list of JSON objects to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    judge_output_dir = Path("/Users/jamiegarnham/lingoly2/judge_output")
    
    # Find all JSONL files in the judge_output directory
    jsonl_files = glob.glob(str(judge_output_dir / "*.jsonl"))
    
    # Group files by model, size, and judge type
    file_groups = defaultdict(lambda: defaultdict(list))
    
    for filepath in jsonl_files:
        filename = Path(filepath).name
        model_family, size, format_name, judge_type = parse_filename(filename)
        
        if model_family != 'unknown' and size and format_name and judge_type:
            key = (model_family, size, judge_type)
            file_groups[key][format_name].append(filepath)
            print(f"Grouped: {filename} -> {key}, format: {format_name}")
    
    # Process each group and combine files
    for (model_family, size, judge_type), format_files in file_groups.items():
        print(f"\nProcessing {model_family} {size} {judge_type}:")
        
        # Skip if this is llama size=2 and we're missing monolingual (which is expected)
        if model_family == 'llama' and size == '2' and 'monolingual' not in format_files:
            print(f"  Note: Skipping monolingual for llama size 2 (expected)")
        
        combined_data = []
        formats_included = []
        
        # Combine data from all available formats
        for format_name in ['match-up', 'monolingual', 'pattern', 'rosetta', 'unknown']:
            if format_name in format_files:
                for filepath in format_files[format_name]:
                    print(f"  Loading {Path(filepath).name}")
                    data = load_jsonl_file(filepath)
                    combined_data.extend(data)
                    formats_included.append(format_name)
        
        # Generate output filename
        output_filename = f"{model_family}_{size}_{judge_type}_combined.jsonl"
        output_path = judge_output_dir / model_family / output_filename
        
        # Save combined file
        print(f"  Saving {len(combined_data)} entries to {output_path}")
        save_jsonl_file(combined_data, output_path)
        print(f"  Formats included: {', '.join(sorted(formats_included))}")

if __name__ == "__main__":
    main()