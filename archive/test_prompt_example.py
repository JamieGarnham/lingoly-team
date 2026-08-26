#!/usr/bin/env python3
"""
Test script to show what the MC judge prompt looks like
"""

import csv
import ast
from llm_mc_responses import extract_subquestion_text, clean_problem_sheet, create_mc_judge_prompt, generate_repeat_answers

def show_prompt_example():
    """Show an example of what the prompt looks like"""
    
    # Load one example from the CSV
    csv_path = "openrouter_analysis/original prompt/subquestion_eval_original_prompt_32.csv"
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv.field_size_limit(1000000)
        reader = csv.DictReader(f)
        
        # Get the first row
        row = next(reader)
        
        # Extract basic info
        print("=== ROW INFO ===")
        print(f"Overall Question N: {row['overall_question_n']}")
        print(f"Question N: {row['question_n']}")
        print(f"Serial: {row['serial']}")
        print(f"Correct Answer: {row['correct_answer']}")
        print(f"Number Correct: {row['number_correct']}")
        print(f"Majority Size: {row['majority_size']}")
        print()
        
        # Get all model answers and parse unique answers
        model_answers = []
        for j in range(1, 33):  # v1 through v32
            answer_key = f'model_answer_v{j}'
            if answer_key in row:
                model_answers.append(row[answer_key])
        
        unique_answers = ast.literal_eval(row['unique_answers'])
        
        # Filter out invalid responses
        filtered_answers = []
        for answer in unique_answers:
            if (answer != "N/A" and 
                not (isinstance(answer, str) and answer.startswith("IMPROPER PARSING:"))):
                filtered_answers.append(answer)
        
        print("=== OPTIONS ===")
        print(f"All unique answers ({len(unique_answers)}): {unique_answers}")
        print(f"Filtered answers ({len(filtered_answers)}): {filtered_answers}")
        print()
        
        # Generate repeat answers
        repeat_answers = generate_repeat_answers(model_answers)
        print(f"Repeat answers ({len(repeat_answers)}): {repeat_answers}")
        print()
        
        # Show the raw problem sheet vs cleaned
        full_problem_sheet = row['questions']
        print("=== RAW PROBLEM SHEET (first 500 chars) ===")
        print(full_problem_sheet[:500] + "..." if len(full_problem_sheet) > 500 else full_problem_sheet)
        print()
        
        cleaned_sheet = clean_problem_sheet(full_problem_sheet)
        print("=== CLEANED PROBLEM SHEET (first 500 chars) ===")
        print(cleaned_sheet[:500] + "..." if len(cleaned_sheet) > 500 else cleaned_sheet)
        print()
        
        # Show extracted subquestion
        subquestion_text = extract_subquestion_text(full_problem_sheet, row['question_n'], row['serial'])
        print("=== EXTRACTED SUBQUESTION ===")
        print(f"'{subquestion_text}'")
        print()
        
        # Create the full prompt using filtered answers as options
        options = filtered_answers[:5] if len(filtered_answers) > 5 else filtered_answers  # Limit to 5 for example
        
        if options:
            prompt = create_mc_judge_prompt(full_problem_sheet, row['question_n'], row['serial'], options)
            
            print("=== FULL MC JUDGE PROMPT ===")
            print(prompt)
            print()
            print(f"Prompt length: {len(prompt)} characters")
        else:
            print("No valid options available for this example")

if __name__ == "__main__":
    show_prompt_example()