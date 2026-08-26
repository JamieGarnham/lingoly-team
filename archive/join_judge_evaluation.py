#!/usr/bin/env python3
"""
Script to join subquestion_eval CSV with judge evaluation results.
Adds judged_answer and is_judge_correct columns.
"""

import csv
import json
import ast
import glob
import sys
from pathlib import Path

# Fix CSV field size limit
csv.field_size_limit(sys.maxsize)

def normalize_answer(answer):
    """
    Normalize an answer for comparison by stripping whitespace and full stops,
    and converting to lowercase. (From evaluate_subquestions.py)
    """
    if not isinstance(answer, str):
        answer = str(answer)
    return answer.strip().rstrip('.').lower()

def parse_correct_answer(answer_value):
    """
    Parse correct answer which could be a string, list, or JSON string.
    (From evaluate_subquestions.py)
    """
    if isinstance(answer_value, list):
        return answer_value
    elif isinstance(answer_value, str):
        # Try to parse as JSON list first
        if answer_value.startswith('[') and answer_value.endswith(']'):
            try:
                return json.loads(answer_value)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(answer_value)
                except:
                    return [answer_value]
        else:
            return [answer_value]
    else:
        return [str(answer_value)]

def check_judge_answer_correctness(judged_answer, correct_answers, overall_question_n=None, question_n=None, serial=None):
    """
    Check if the judged answer matches any of the correct answers.
    Uses the same logic as evaluate_subquestions.py
    Includes special handling for specific questions.
    """
    if not judged_answer or judged_answer.strip() == "":
        return False
    
    # Special case for question 5, Q 5.1, serial a
    if (overall_question_n == 5 and question_n == "Q 5.1" and serial == "a"):
        normalized_answer = normalize_answer(judged_answer)
        # Check if answer contains both "üpgontüd" and "sopostüd"
        if "üpgontüd" in normalized_answer and "sopostüd" in normalized_answer:
            return True
    
    # Special case for question 170, Q5., k - add "langgbu'" and "maysu'" as correct
    if (overall_question_n == 170 and question_n == "Q 5." and serial == "k"):
        normalized_answer = normalize_answer(judged_answer)
        # Check for langgbu or maysu with either apostrophe type (' or ')
        straight_apos = "'"  # ASCII 39
        curly_apos = chr(8217)  # Unicode 8217 '
        if (normalized_answer == f"langgbu{straight_apos}" or normalized_answer == f"langgbu{curly_apos}" or
            normalized_answer == f"maysu{straight_apos}" or normalized_answer == f"maysu{curly_apos}"):
            return True
    
    # Special case for question 75, Q7., 3 - add "two people who are not siblings"
    if (overall_question_n == 75 and question_n == "Q 7." and serial == "3"):
        normalized_answer = normalize_answer(judged_answer)
        if "two people who are not siblings" in normalized_answer:
            return True
    
    # Standard correctness check
    normalized_judged = normalize_answer(judged_answer)
    
    # Parse correct answers if needed
    if isinstance(correct_answers, str):
        correct_answers_list = parse_correct_answer(correct_answers)
    else:
        correct_answers_list = correct_answers
    
    normalized_correct = [normalize_answer(ca) for ca in correct_answers_list]
    
    return normalized_judged in normalized_correct

def classify_judge_validity(scores):
    """
    Classify judge validity based on the scores pattern.
    
    Args:
        scores: List of numeric scores or None if not available
        
    Returns:
        str: "reranker", "scorer", or "invalid"
    """
    if not scores or not isinstance(scores, list) or len(scores) == 0:
        return "invalid"
    
    try:
        # Convert to float and check for parsing issues
        numeric_scores = []
        for score in scores:
            if score is None or score == "":
                return "invalid"
            numeric_scores.append(float(score))
        
        # Check if all scores are 0.0
        if all(score == 0.0 for score in numeric_scores):
            return "invalid"
        
        # Check for reranker pattern: positive whole numbers with one equal to 1.0
        if all(score > 0 and score == int(score) for score in numeric_scores):
            if numeric_scores.count(1.0) == 1:
                return "reranker"
            else:
                return "invalid"
        
        # Check for scorer pattern: all between 0 and 1 inclusive with singular maximum
        if all(0.0 <= score <= 1.0 for score in numeric_scores):
            max_score = max(numeric_scores)
            max_count = numeric_scores.count(max_score)
            if max_count == 1:
                return "scorer"
            else:
                return "invalid"
        
        # Any other pattern is invalid
        return "invalid"
        
    except (ValueError, TypeError):
        return "invalid"

def load_judge_evaluation(judge_file_path):
    """
    Load the judge evaluation JSONL and create a lookup dictionary.
    Key: (overall_question_n, question_n, serial)
    Value: (judged_answer, judge_prompt, scores)
    """
    judge_lookup = {}
    
    # Check if it's JSONL or CSV
    if judge_file_path.endswith('.jsonl'):
        # Load JSONL format (regular judge output)
        with open(judge_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    key = (
                        int(row['overall_question_n']),
                        row['question_n'].strip(),
                        row['serial'].strip()
                    )
                    
                    # For regular judge files, extract judged answer based on scores
                    judged_answer = ""
                    scores = None
                    
                    # Extract scores from the judge response
                    if 'scores' in row:
                        scores = row['scores']
                    
                    # Extract judged answer based on judge validity
                    if scores and isinstance(scores, dict) and 'unique_answers' in row:
                        try:
                            score_values = list(scores.values())
                            judge_validity = classify_judge_validity(score_values)
                            unique_answers = row['unique_answers']
                            
                            answer_index = None
                            if judge_validity == "reranker":
                                # Find the answer with score = 1.0
                                for answer_key, score in scores.items():
                                    if score == 1.0:
                                        if answer_key.startswith('answer_'):
                                            answer_index = int(answer_key.replace('answer_', '')) - 1  # Convert to 0-based index
                                        break
                            elif judge_validity == "scorer":
                                # Find the answer with the maximum score
                                max_score = max(scores.values())
                                for answer_key, score in scores.items():
                                    if score == max_score:
                                        if answer_key.startswith('answer_'):
                                            answer_index = int(answer_key.replace('answer_', '')) - 1  # Convert to 0-based index
                                        break
                            
                            # Get the actual answer text from unique_answers
                            if answer_index is not None and 0 <= answer_index < len(unique_answers):
                                judged_answer = unique_answers[answer_index]
                        except:
                            judged_answer = ""
                    
                    judge_lookup[key] = (judged_answer, row.get('judge_prompt', ''), scores)
    else:
        # Load CSV format (original judge output)
        with open(judge_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    int(row['overall_question_n']),
                    row['question_n'].strip(),
                    row['serial'].strip()
                )
                # Extract scores from CSV if available
                scores = None
                if 'scores' in row:
                    try:
                        scores = json.loads(row['scores']) if row['scores'] else None
                    except:
                        scores = None
                
                judge_lookup[key] = (row['judged_answer'].strip(), row['judge_prompt'].strip(), scores)
    
    print(f"Loaded {len(judge_lookup)} judge evaluations")
    return judge_lookup

def load_mc_judge_evaluation(mc_judge_file_path):
    """
    Load the MC judge evaluation JSONL and create a lookup dictionary.
    Key: (overall_question_n, question_n, serial)
    Value: (mc_judged_answer, mc_judge_prompt, mc_judge_validity)
    """
    mc_judge_lookup = {}
    
    with open(mc_judge_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                key = (
                    int(row['overall_question_n']),
                    row['question_n'].strip(),
                    row['serial'].strip()
                )
                
                # Extract MC judged answer and check validity
                mc_judged_answer = ""
                mc_judge_validity = False
                
                if 'parse_result' in row and row['parse_result'].get('valid', False):
                    mc_judged_answer = row['parse_result'].get('selected_answer', '')
                    mc_judge_validity = True
                
                mc_judge_lookup[key] = (mc_judged_answer, row.get('mc_judge_prompt', ''), mc_judge_validity)
    
    print(f"Loaded {len(mc_judge_lookup)} MC judge evaluations")
    return mc_judge_lookup

def find_matching_judge_files(base_filename):
    """
    Find matching judge and mc_judge files based on a base filename.
    Returns (judge_file_path, mc_judge_file_path) or (None, None) if not found.
    """
    from pathlib import Path
    import glob
    
    judge_output_dir = Path("/Users/jamiegarnham/lingoly2/judge_output")
    
    # Extract the pattern from base filename (remove extension and path)
    base_name = Path(base_filename).stem
    
    # Look for judge files (not mc_judge)
    judge_pattern = str(judge_output_dir / f"*judge_evaluation*.jsonl")
    mc_judge_pattern = str(judge_output_dir / f"*mc_judge_evaluation*.jsonl")
    
    judge_files = glob.glob(judge_pattern)
    mc_judge_files = glob.glob(mc_judge_pattern)
    
    # Filter out mc_judge files from judge_files
    judge_files = [f for f in judge_files if "mc_judge" not in f]
    
    # Try to find matching files based on common patterns
    best_judge = None
    best_mc_judge = None
    
    # Look for files that share the most common elements with the base name
    for judge_file in judge_files:
        judge_name = Path(judge_file).stem
        # Remove timestamps and common suffixes to get the core pattern
        judge_core = judge_name.split('_judge_evaluation_')[0] if '_judge_evaluation_' in judge_name else judge_name
        
        for mc_judge_file in mc_judge_files:
            mc_judge_name = Path(mc_judge_file).stem
            mc_judge_core = mc_judge_name.split('_mc_judge_evaluation_')[0] if '_mc_judge_evaluation_' in mc_judge_name else mc_judge_name
            
            if judge_core == mc_judge_core:
                best_judge = judge_file
                best_mc_judge = mc_judge_file
                break
        
        if best_judge:
            break
    
    return best_judge, best_mc_judge

def join_with_judge_evaluation(subquestion_csv_path, judge_csv_path=None, output_csv_path=None, mc_judge_csv_path=None):
    """
    Join the subquestion evaluation CSV with both judge and MC judge evaluation data.
    If judge_csv_path is not provided, will attempt to find matching files automatically.
    """
    
    # If no judge files specified, try to find them automatically
    if judge_csv_path is None:
        judge_csv_path, mc_judge_csv_path = find_matching_judge_files(subquestion_csv_path)
        if not judge_csv_path or not mc_judge_csv_path:
            print("Could not find matching judge files automatically.")
            print("Please provide judge file paths manually.")
            return
    else:
        # If judge path provided but mc_judge_csv_path not provided, try to find corresponding MC judge path
        if mc_judge_csv_path is None:
            mc_judge_csv_path = judge_csv_path.replace('_judge_evaluation_', '_mc_judge_evaluation_')
            if not Path(mc_judge_csv_path).exists():
                print(f"Could not find corresponding MC judge file: {mc_judge_csv_path}")
                return
    
    print(f"Using judge file: {judge_csv_path}")
    print(f"Using MC judge file: {mc_judge_csv_path}")
    
    # Load both judge evaluation lookups
    judge_lookup = load_judge_evaluation(judge_csv_path)
    mc_judge_lookup = load_mc_judge_evaluation(mc_judge_csv_path)
    
    
    # Process subquestion CSV
    rows_processed = 0
    judge_matches = 0
    mc_judge_matches = 0
    fallback_used = 0
    
    with open(subquestion_csv_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Get all existing fieldnames and add new ones
        fieldnames = reader.fieldnames + [
            'judged_answer', 'judge_prompt', 'is_judge_correct', 'judge_validity', 'is_judge_used',
            'mc_judged_answer', 'mc_judge_prompt', 'is_mc_judge_correct', 'mc_judge_validity', 'is_mc_judge_used'
        ]
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                rows_processed += 1
                
                # Create lookup key
                key = (
                    int(row['overall_question_n']),
                    row['question_n'].strip(),
                    row['serial'].strip()
                )
                
                
                # Process regular judge evaluation
                if key in judge_lookup:
                    judged_answer, judge_prompt, scores = judge_lookup[key]
                    row['judged_answer'] = judged_answer
                    row['judge_prompt'] = judge_prompt
                    row['is_judge_used'] = True
                    judge_matches += 1
                    
                    # Classify judge validity based on scores
                    if scores and isinstance(scores, dict):
                        score_values = list(scores.values())
                        row['judge_validity'] = classify_judge_validity(score_values)
                    else:
                        row['judge_validity'] = "invalid"
                    
                    # Check if judge answer is correct
                    if row['judge_validity'] == "invalid":
                        # Invalid judge results should always be marked as incorrect
                        row['is_judge_correct'] = False
                    elif judged_answer:
                        is_correct = check_judge_answer_correctness(judged_answer, row['correct_answer'], key[0], key[1], key[2])
                        row['is_judge_correct'] = is_correct
                    else:
                        # Empty judged_answer, mark as incorrect
                        row['is_judge_correct'] = False
                else:
                    # No judge evaluation found, use fallback
                    row['judged_answer'] = ""
                    row['judge_prompt'] = ""
                    row['is_judge_correct'] = ""
                    row['judge_validity'] = ""
                    row['is_judge_used'] = False
                    fallback_used += 1
                
                # Process MC judge evaluation
                if key in mc_judge_lookup:
                    mc_judged_answer, mc_judge_prompt, mc_judge_validity = mc_judge_lookup[key]
                    row['mc_judged_answer'] = mc_judged_answer
                    row['mc_judge_prompt'] = mc_judge_prompt
                    row['mc_judge_validity'] = mc_judge_validity
                    row['is_mc_judge_used'] = True
                    
                    # Check if MC judge answer is correct
                    if mc_judge_validity == False:
                        # Invalid MC judge results should always be marked as incorrect
                        row['is_mc_judge_correct'] = False
                    elif mc_judged_answer and mc_judge_validity:
                        is_mc_correct = check_judge_answer_correctness(mc_judged_answer, row['correct_answer'], key[0], key[1], key[2])
                        row['is_mc_judge_correct'] = is_mc_correct
                        mc_judge_matches += 1
                    else:
                        # Empty MC judged_answer, mark as incorrect
                        row['is_mc_judge_correct'] = False
                else:
                    # No MC judge evaluation found, use fallback
                    row['mc_judged_answer'] = ""
                    row['mc_judge_prompt'] = ""
                    row['is_mc_judge_correct'] = ""
                    row['mc_judge_validity'] = ""
                    row['is_mc_judge_used'] = False
                
                writer.writerow(row)
    
    print(f"Processed {rows_processed} total rows")
    print(f"Found judge evaluations for {judge_matches} rows")
    print(f"Found MC judge evaluations for {mc_judge_matches} rows")
    print(f"Used fallback (is_any_correct) for {fallback_used} rows")
    print(f"Output saved to: {output_csv_path}")

def find_files_by_model_and_size(model_type, sample_size):
    """
    Find the appropriate subquestion_eval, judge, and mc_judge files based on model type and sample size.
    
    Args:
        model_type: One of 'deepseek', 'gemini', 'llama'
        sample_size: One of 32, 16, 8, 4, 2, 1
        
    Returns:
        tuple: (subquestion_file, judge_file, mc_judge_file)
    """
    base_dir = Path("/Users/jamiegarnham/lingoly2")
    
    # Validate inputs
    valid_models = ['deepseek', 'gemini', 'llama']
    valid_sizes = [32, 16, 8, 4, 2, 1]
    
    if model_type not in valid_models:
        raise ValueError(f"Model type must be one of: {valid_models}")
    
    if sample_size not in valid_sizes:
        raise ValueError(f"Sample size must be one of: {valid_sizes}")
    
    # Construct file paths
    subquestion_file = base_dir / "openrouter_analysis" / model_type / f"subquestion_eval_{model_type}_shuffle_{sample_size}_fix.csv"
    judge_file = base_dir / "judge_output" / model_type / f"{model_type}_{sample_size}_judge_combined.jsonl"
    mc_judge_file = base_dir / "judge_output" / model_type / f"{model_type}_{sample_size}_mc_judge_combined.jsonl"
    
    # Verify files exist
    missing_files = []
    if not subquestion_file.exists():
        missing_files.append(str(subquestion_file))
    if not judge_file.exists():
        missing_files.append(str(judge_file))
    if not mc_judge_file.exists():
        missing_files.append(str(mc_judge_file))
    
    if missing_files:
        raise FileNotFoundError(f"Required files not found: {missing_files}")
    
    return str(subquestion_file), str(judge_file), str(mc_judge_file)

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        # New usage: python join_judge_evaluation.py <model_type> <sample_size>
        try:
            model_type = sys.argv[1].lower()
            sample_size = int(sys.argv[2])
            
            # Find files automatically
            subquestion_file, judge_file, mc_judge_file = find_files_by_model_and_size(model_type, sample_size)
            
            # Generate output filename in the all_judge_mc_judge_evals directory
            output_dir = Path("/Users/jamiegarnham/lingoly2/openrouter_analysis/all_judge_mc_judge_evals")
            output_dir.mkdir(exist_ok=True)
            output_file = str(output_dir / f"subquestion_eval_{model_type}_shuffle_{sample_size}_fix_with_judge_evaluation.csv")
            
            print(f"Using files:")
            print(f"  Subquestion eval: {subquestion_file}")
            print(f"  Judge file: {judge_file}")
            print(f"  MC Judge file: {mc_judge_file}")
            print(f"  Output file: {output_file}")
            
            # Set variables for function call
            judge_csv_path = judge_file
            mc_judge_csv_path = mc_judge_file
            
        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
            print("Usage: python join_judge_evaluation.py <model_type> <sample_size>")
            print("  model_type: deepseek, gemini, or llama")
            print("  sample_size: 32, 16, 8, 4, 2, or 1")
            sys.exit(1)
            
    elif len(sys.argv) >= 2:
        # Legacy usage: python join_judge_evaluation.py <subquestion_file> [judge_file] [output_file]
        subquestion_file = sys.argv[1]
        judge_csv_path = sys.argv[2] if len(sys.argv) >= 3 else None
        output_file = sys.argv[3] if len(sys.argv) >= 4 else None
        
        # Set default output file if not provided
        if output_file is None:
            base_name = Path(subquestion_file).stem
            output_file = str(Path(subquestion_file).parent / f"{base_name}_with_judge_evaluation.csv")
        
        # Ensure subquestion file exists
        if not Path(subquestion_file).exists():
            raise FileNotFoundError(f"Required file not found: {subquestion_file}")
            
        # Use modified function to handle both judge types
        mc_judge_csv_path = judge_csv_path.replace('_judge_evaluation_', '_mc_judge_evaluation_') if judge_csv_path else None
        if mc_judge_csv_path and not Path(mc_judge_csv_path).exists():
            print(f"Warning: Could not find corresponding MC judge file: {mc_judge_csv_path}")
            mc_judge_csv_path = None
            
    else:
        print("Usage: python join_judge_evaluation.py <model_type> <sample_size>")
        print("  model_type: deepseek, gemini, or llama")
        print("  sample_size: 32, 16, 8, 4, 2, or 1")
        print()
        print("Alternative legacy usage: python join_judge_evaluation.py <subquestion_file> [judge_file] [output_file]")
        sys.exit(1)
    
    print("Starting join operation...")
    join_with_judge_evaluation(subquestion_file, judge_csv_path, output_file, mc_judge_csv_path)
    print("Done!")