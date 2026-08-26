#!/usr/bin/env python3
"""
Comprehensive test script to validate MC judge prompt creation on multiple examples
"""

import csv
import ast
from llm_mc_responses import extract_subquestion_text, clean_problem_sheet, create_mc_judge_prompt

def test_subquestion_extraction():
    """Test subquestion extraction on multiple examples from the CSV"""
    
    csv_path = "openrouter_analysis/original_prompt/subquestion_eval_original_prompt_32.csv"
    
    test_cases = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv.field_size_limit(1000000)
        reader = csv.DictReader(f)
        
        # Collect diverse examples - first 20 rows to get variety
        for i, row in enumerate(reader):
            if i >= 20:  # Test first 20 examples
                break
                
            test_cases.append({
                'row_num': i + 1,
                'overall_question_n': row['overall_question_n'],
                'question_n': row['question_n'],
                'serial': row['serial'],
                'questions': row['questions'],
                'correct_answer': row['correct_answer']
            })
    
    print(f"Testing subquestion extraction on {len(test_cases)} examples...")
    print("=" * 80)
    
    success_count = 0
    issues = []
    
    for case in test_cases:
        print(f"\n--- Test Case {case['row_num']} ---")
        print(f"Question: {case['question_n']}, Serial: {case['serial']}")
        
        # Extract subquestion
        extracted = extract_subquestion_text(case['questions'], case['question_n'], case['serial'])
        
        print(f"Extracted: '{extracted}'")
        
        # Check if extraction looks reasonable
        is_success = True
        if extracted == f"Answer question {case['serial']}.":
            print("⚠️  WARNING: Fell back to generic answer")
            is_success = False
            issues.append(f"Row {case['row_num']}: Generic fallback for {case['question_n']}/{case['serial']}")
        elif len(extracted) < 10:
            print("⚠️  WARNING: Very short extraction")
            is_success = False
            issues.append(f"Row {case['row_num']}: Short extraction: '{extracted}'")
        else:
            print("✅ Extraction looks good")
        
        if is_success:
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {success_count}/{len(test_cases)} successful extractions")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    
    return test_cases[:3]  # Return first 3 for detailed prompt testing

def test_full_prompt_creation(test_cases):
    """Test full prompt creation on a few examples"""
    
    print("\n" + "=" * 80)
    print("TESTING FULL PROMPT CREATION")
    print("=" * 80)
    
    for case in test_cases:
        print(f"\n--- Full Prompt for Row {case['row_num']} ---")
        print(f"Question: {case['question_n']}, Serial: {case['serial']}")
        
        # Create dummy options for testing
        dummy_options = ["option1", "option2", "option3"]
        
        try:
            prompt = create_mc_judge_prompt(case['questions'], case['question_n'], case['serial'], dummy_options)
            
            print(f"Prompt length: {len(prompt)} characters")
            
            # Show key parts of the prompt
            lines = prompt.split('\n')
            print("\nPrompt structure:")
            print(f"  - Opening: {lines[0][:60]}...")
            
            # Find the subquestion line
            for i, line in enumerate(lines):
                if line.startswith(case['serial'] + ' '):
                    print(f"  - Subquestion: {line}")
                    break
            
            # Find options section
            for i, line in enumerate(lines):
                if "These are the options you have to choose from:" in line:
                    print(f"  - Options section found at line {i}")
                    break
            
            print("✅ Prompt created successfully")
            
        except Exception as e:
            print(f"❌ ERROR creating prompt: {e}")

if __name__ == "__main__":
    # Test subquestion extraction
    sample_cases = test_subquestion_extraction()
    
    # Test full prompt creation on a few examples
    test_full_prompt_creation(sample_cases)