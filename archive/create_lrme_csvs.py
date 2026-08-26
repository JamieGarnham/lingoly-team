#!/usr/bin/env python3
"""
Script to convert chained subquestion evaluation CSVs to LRME format for R analysis.
Adds required columns for Mixed Effects Logistic Regression analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

def create_lrme_format(input_csv_path, output_csv_path, model_name, format_name):
    """
    Convert a chained subquestion CSV to LRME format for R analysis.
    """
    print(f"Processing {model_name} {format_name}...")
    
    # Read the input CSV
    df = pd.read_csv(input_csv_path)
    
    # Create lookup column (combination of overall_question_n and question_n)
    df['lookup'] = df['overall_question_n'].astype(str) + df['question_n'].astype(str)
    
    # Sort data by overall_question_n, then question_n, then serial
    df = df.sort_values(['overall_question_n', 'question_n', 'serial']).reset_index(drop=True)
    
    # Identify first row for each overall_question_n
    # Mark is_first as True only for the first row of each overall_question_n
    df['is_first'] = False
    first_rows = df.groupby('overall_question_n').head(1).index
    df.loc[first_rows, 'is_first'] = True
    
    # Create a mapping of first subquestion results for each question and version
    first_subq_results = df[df['is_first']].copy()
    
    # Extract version numbers from column names
    version_cols = [col for col in df.columns if re.match(r'is_v\d+_correct', col)]
    versions = [int(re.findall(r'v(\d+)_correct', col)[0]) for col in version_cols]
    
    # Create vN_is_first_correct columns
    for version in sorted(versions):
        col_name = f'v{version}_is_first_correct'
        df[col_name] = False  # Default to False
        
        # For each row, find if this version got the first subquestion correct for this overall_question_n
        for idx, row in df.iterrows():
            # Find the first subquestion result for this overall_question_n and version
            first_result = first_subq_results[
                first_subq_results['overall_question_n'] == row['overall_question_n']
            ]
            
            if not first_result.empty:
                first_correct = first_result.iloc[0][f'is_v{version}_correct']
                df.at[idx, col_name] = first_correct
    
    # Calculate summary statistics
    # For each row (subquestion), calculate statistics about first subquestion performance
    summary_stats = []
    
    for idx, row in df.iterrows():
        # Get first subquestion results for this overall_question_n
        first_result = first_subq_results[
            first_subq_results['overall_question_n'] == row['overall_question_n']
        ]
        
        if first_result.empty:
            # No first subquestion found, use defaults
            stats = {
                'num_first_correct': 0,
                'num_first_incorrect': 0,
                'num_correct_when_first_correct': 0,
                'num_correct_when_first_incorrect': 0,
                'pct_correct_when_first_correct': None,
                'pct_correct_when_first_incorrect': None
            }
        else:
            # Count how many versions got first subquestion correct/incorrect
            first_row = first_result.iloc[0]
            first_correct_count = sum(first_row[f'is_v{v}_correct'] for v in versions)
            first_incorrect_count = len(versions) - first_correct_count
            
            # For current subquestion, count how many got it right when first was right/wrong
            current_correct_when_first_correct = 0
            current_correct_when_first_incorrect = 0
            
            for version in versions:
                first_correct = first_row[f'is_v{version}_correct']
                current_correct = row[f'is_v{version}_correct']
                
                if first_correct and current_correct:
                    current_correct_when_first_correct += 1
                elif not first_correct and current_correct:
                    current_correct_when_first_incorrect += 1
            
            # Calculate percentages (use None when denominator is 0)
            pct_correct_when_first_correct = (current_correct_when_first_correct / first_correct_count) if first_correct_count > 0 else None
            pct_correct_when_first_incorrect = (current_correct_when_first_incorrect / first_incorrect_count) if first_incorrect_count > 0 else None
            
            stats = {
                'num_first_correct': first_correct_count,
                'num_first_incorrect': first_incorrect_count,
                'num_correct_when_first_correct': current_correct_when_first_correct,
                'num_correct_when_first_incorrect': current_correct_when_first_incorrect,
                'pct_correct_when_first_correct': pct_correct_when_first_correct,
                'pct_correct_when_first_incorrect': pct_correct_when_first_incorrect
            }
        
        summary_stats.append(stats)
    
    # Add summary statistics to dataframe
    for key in summary_stats[0].keys():
        df[key] = [stat[key] for stat in summary_stats]
    
    # Add multi question indicator (True if the question has multiple subquestions)
    question_subq_counts = df.groupby(['overall_question_n', 'question_n']).size().reset_index(name='subq_count')
    question_subq_counts['multi question'] = question_subq_counts['subq_count'] > 1
    df = df.merge(question_subq_counts[['overall_question_n', 'question_n', 'multi question']], 
                  on=['overall_question_n', 'question_n'], how='left')
    
    # Reorder columns to match the original LRME format
    # Start with basic columns
    base_cols = ['questions', 'overall_question_n', 'question_n', 'lookup', 'serial', 'format', 'format_fixed', 'correct_answer']
    
    # Add model answer and correctness columns in alternating pattern
    model_cols = []
    for version in sorted(versions):
        model_cols.extend([f'model_answer_v{version}', f'is_v{version}_correct', f'v{version}_is_first_correct'])
    
    # Add analysis columns
    analysis_cols = ['majority', 'tiebreaker', 'majority_size', 'unique_answers', 
                    'is_majority_correct', 'is_tiebreaker_correct', 'is_any_correct', 'number_correct']
    
    # Add LRME-specific columns
    lrme_cols = ['is_first', 'num_first_correct', 'num_first_incorrect', 
                'num_correct_when_first_correct', 'num_correct_when_first_incorrect',
                'pct_correct_when_first_correct', 'pct_correct_when_first_incorrect', 'multi question']
    
    # Combine all columns
    final_cols = base_cols + model_cols + analysis_cols + lrme_cols
    
    # Select only columns that exist in the dataframe
    available_cols = [col for col in final_cols if col in df.columns]
    df_final = df[available_cols]
    
    # Clean up any helper columns (none needed with new approach)
    if 'subq_count' in df_final.columns:
        df_final = df_final.drop('subq_count', axis=1)
    
    # Save to output file
    df_final.to_csv(output_csv_path, index=False)
    print(f"Saved LRME format CSV: {output_csv_path}")
    print(f"  Rows: {len(df_final)}")
    print(f"  Columns: {len(df_final.columns)}")
    print(f"  Multi-part questions: {df_final['multi question'].sum()}")
    print(f"  First subquestions: {df_final['is_first'].sum()}")
    print()

def extract_format_from_all_formats_csv(input_csv_path, format_name):
    """
    Extract a specific format from the all_formats CSV.
    """
    df = pd.read_csv(input_csv_path)
    
    # Filter by format_fixed column
    format_df = df[df['format_fixed'] == format_name.title()].copy()
    
    return format_df

def main():
    """
    Process all model/format combinations to create LRME CSVs.
    """
    models = ['llama', 'gemini', 'deepseek']
    formats = ['rosetta', 'monolingual', 'pattern']
    
    base_dir = Path('openrouter_analysis')
    
    for model in models:
        model_dir = base_dir / model
        
        # Read the all_formats CSV for this model
        all_formats_csv = model_dir / f'chained_subquestion_eval_{model}_all_formats_32.csv'
        
        if not all_formats_csv.exists():
            print(f"Warning: {all_formats_csv} not found. Skipping {model}.")
            continue
        
        print(f"Processing {model} model...")
        
        for format_name in formats:
            # Extract format-specific data
            df_all = pd.read_csv(all_formats_csv)
            df_format = df_all[df_all['format_fixed'] == format_name.title()].copy()
            
            if df_format.empty:
                print(f"Warning: No {format_name} data found for {model}. Skipping.")
                continue
            
            # Create temporary CSV for this format
            temp_csv = f'temp_{model}_{format_name}_32.csv'
            df_format.to_csv(temp_csv, index=False)
            
            # Output filename
            output_csv = f'chained_{format_name}_LRME_32_{model}.csv'
            
            # Process to LRME format
            create_lrme_format(temp_csv, output_csv, model, format_name)
            
            # Clean up temporary file
            Path(temp_csv).unlink()
    
    print("All LRME CSV files created successfully!")

if __name__ == "__main__":
    main()