#!/usr/bin/env python3
"""
Generate CSV of all prompts that would be created by the MC judge pipeline
without actually calling the LLM API
"""

import csv
import json
import os
import ast
from datetime import datetime
from pathlib import Path

# Import functions from the main script
from llm_mc_responses import (
    load_evaluation_csv, 
    create_mc_judge_prompt,
    generate_repeat_answers,
    get_answer_frequencies,
    get_top_frequent_answers
)

def generate_prompts_csv(
    response_data_csv: str = "openrouter_analysis/original_prompt/subquestion_eval_original_prompt_32.csv",
    output_csv: str = "mc_judge_prompts.csv",
    question_filter: int = None,
    serial_filter: str = None,
    limit: int = None,
    only_any_correct: bool = False,
    min_unique_answers: int = 1,
    repeat_answers_only: bool = False,
    max_options: int = None,
    majority_threshold: float = None,
):
    """Generate CSV of all prompts without calling LLM API"""
    
    print(f"Generating prompts CSV with parameters:")
    print(f"  - response_data_csv: {response_data_csv}")
    print(f"  - only_any_correct: {only_any_correct}")
    print(f"  - min_unique_answers: {min_unique_answers}")
    print(f"  - repeat_answers_only: {repeat_answers_only}")
    print(f"  - majority_threshold: {majority_threshold}")
    print(f"  - max_options: {max_options}")
    print()
    
    # Load evaluation data with the same filters as the main pipeline
    print(f"Loading evaluation data from {response_data_csv}")
    rows = load_evaluation_csv(
        response_data_csv, 
        question_filter=question_filter,
        serial_filter=serial_filter,
        limit=limit,
        only_any_correct=only_any_correct,
        min_unique_answers=min_unique_answers,
        repeat_answers_only=repeat_answers_only,
        majority_threshold=majority_threshold
    )
    
    if len(rows) == 0:
        print("No rows to process. Exiting.")
        return
    
    print(f"Processing {len(rows)} rows...")
    
    # Prepare CSV output
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV fieldnames
    fieldnames = [
        'row_id',
        'overall_question_n',
        'question_n', 
        'serial',
        'correct_answer',
        'number_correct',
        'majority_size',
        'is_any_correct',
        'unique_answers_count',
        'repeat_answers_count',
        'options_used',
        'options_count',
        'repeat_answers_only_flag',
        'max_options_used',
        'prompt',
        'prompt_length',
        'timestamp'
    ]
    
    results = []
    
    for i, row in enumerate(rows):
        print(f"Processing row {i+1}/{len(rows)}: {row['overall_question_n']}_{row['question_n']}_{row['serial']}")
        
        row_id = f"{row['overall_question_n']}_{row['question_n']}_{row['serial']}"
        
        # Determine which answers to use as options
        if repeat_answers_only:
            options = row['repeat_answers']
        else:
            options = row['unique_answers_parsed']
        
        # Apply max_options filtering if specified
        if max_options and len(options) > max_options:
            # Get top frequent answers
            options = get_top_frequent_answers(row['answer_frequencies'], max_options)
        
        # Skip if no options available
        if len(options) == 0:
            print(f"  - Skipping {row_id}: No options available")
            continue
        
        # Get full problem sheet
        full_problem_sheet = row['questions']
        
        # Create multiple choice judge prompt
        try:
            mc_judge_prompt = create_mc_judge_prompt(full_problem_sheet, row['question_n'], row['serial'], options)
            
            # Create result entry
            result = {
                'row_id': row_id,
                'overall_question_n': int(row['overall_question_n']),
                'question_n': row['question_n'],
                'serial': row['serial'],
                'correct_answer': row['correct_answer'],
                'number_correct': int(row['number_correct']),
                'majority_size': int(row['majority_size']) if 'majority_size' in row else '',
                'is_any_correct': row['is_any_correct'],
                'unique_answers_count': len(row['unique_answers_parsed']),
                'repeat_answers_count': len(row['repeat_answers']),
                'options_used': json.dumps(options),
                'options_count': len(options),
                'repeat_answers_only_flag': repeat_answers_only,
                'max_options_used': max_options,
                'prompt': mc_judge_prompt,
                'prompt_length': len(mc_judge_prompt),
                'timestamp': datetime.now().isoformat(),
            }
            
            results.append(result)
            print(f"  - Generated prompt ({len(mc_judge_prompt)} chars, {len(options)} options)")
            
        except Exception as e:
            print(f"  - ERROR generating prompt for {row_id}: {str(e)}")
            continue
    
    # Write CSV
    print(f"\nWriting {len(results)} prompts to {output_path}")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result)
    
    print(f"CSV generated successfully!")
    print(f"Total prompts: {len(results)}")
    
    # Print some statistics
    if results:
        total_prompt_chars = sum(r['prompt_length'] for r in results)
        avg_prompt_length = total_prompt_chars // len(results)
        option_counts = [r['options_count'] for r in results]
        avg_options = sum(option_counts) // len(option_counts)
        
        print(f"\nStatistics:")
        print(f"  - Average prompt length: {avg_prompt_length} characters")
        print(f"  - Average options per prompt: {avg_options}")
        print(f"  - Min options: {min(option_counts)}")
        print(f"  - Max options: {max(option_counts)}")
        print(f"  - Total characters: {total_prompt_chars:,}")
    
    return output_path

if __name__ == "__main__":
    import fire
    
    # Set default parameters to match the command you want to test
    fire.Fire(generate_prompts_csv)