#!/usr/bin/env python3
"""
Chained prompting approach for Match-up problems from benchmark_same_obf dataset.

This script implements a two-stage approach:
1. First prompt: Ask the model to analyze the language and determine patterns
2. Second+ prompts: Ask specific questions based on the analysis

Output format matches existing evaluation scripts with added 'reasoning_output' field.
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import fire
from tqdm import tqdm

# Add current directory to path for importing prompt_models (now in same folder)
sys.path.append('.')
try:
    import prompt_models
except ImportError:
    print("Error: Could not import prompt_models. Make sure you're running from the testing/code directory.")
    sys.exit(1)

# Constants
MAX_API_ATTEMPT = 10
# Dynamic path detection
script_dir = os.path.dirname(os.path.abspath(__file__))
if "testing_" in script_dir:
    # Running from testing_X/code directory
    base_dir = os.path.join(script_dir, "..", "..", "testing")
else:
    # Running from testing/code directory
    base_dir = os.path.join(script_dir, "..")
OUTFOLDER = os.path.join(base_dir, "data", "chained_responses")
TMP_PATH = os.path.join(base_dir, "data", "chained_responses", "tmp")
MODEL_LIST = os.path.join(base_dir, "data", "model_list.json")
def load_questions(file_path: str) -> Dict[int, List[Dict]]:
    """Load questions grouped by overall_question_n"""
    questions_by_problem = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            overall_question_n = data['index'][0]
            questions_by_problem[overall_question_n].append(data)
    
    # Sort questions within each problem by question number
    for problem_n in questions_by_problem:
        questions_by_problem[problem_n].sort(key=lambda x: x['question_details']['question_n'])
    
    return questions_by_problem

def create_full_problem_sheet(problem_data: List[Dict]) -> str:
    """Create the full problem sheet from metadata"""
    first_question = problem_data[0]
    metadata = first_question['question_details']['metadata']
    
    preamble = metadata.get('preamble', '')
    context = metadata.get('context', '')
    
    # Construct full problem sheet similar to load_questions.py
    problem_sheet = f"""{preamble}
{context}"""
    
    return problem_sheet

def create_initial_matching_prompt(problem_data: List[Dict]) -> str:
    """Create the initial prompt for iterative matching"""
    problem_sheet = create_full_problem_sheet(problem_data)
    
    prompt = f"""Below is a problem sheet from a linguistics exam. Your answers to the questions should rely only on reasoning about the information provided in the sheet.

<START OF PROBLEM SHEET>
{problem_sheet}
<END OF PROBLEM SHEET>

Your task is to:
1. Determine which pair is the MOST LIKELY first pair to match-up.
2. Express this match-up using the following JSON: {{"%%": "X"}}
where %% is the serial, and X is the corresponding translation.
Do not match-up any other pairs yet."""
    
    return prompt

def create_followup_question_prompt(question_data: Dict) -> str:
    """Create a prompt for follow-up questions after matching"""
    prompt = question_data['question_details']['prompt']
    subprompts = question_data['question_details']['subprompts']
    
    # Create the question text
    question_text = f"Based on the linguistic patterns and correspondences you have identified, answer the following question:\n{prompt}"
    
    # Add subquestion details if present
    if subprompts and len(subprompts) > 0:
        if len(subprompts) == 1 and subprompts[0]['question'] == '':
            # Single question with no separate parts
            pass
        else:
            # Multiple parts
            for subprompt in subprompts:
                if subprompt['question']:
                    question_text += f"\n{subprompt['questionpart_n']} {subprompt['question']}"
    
    # Create JSON output format
    json_keys = []
    for subprompt in subprompts:
        json_keys.append(f'"{subprompt["questionpart_n"]}": ""')
    
    json_format = "{" + ", ".join(json_keys) + "}"
    
    question_text += f"\nProvide your answer as json output with the keys as provided below:\n{json_format}"
    
    return question_text

def create_next_matching_prompt(confirmed_matches: List[tuple]) -> str:
    """Create prompt for the next match in the sequence"""
    match_text = "\n".join([f"{serial} matches up to {translation}" for serial, translation in confirmed_matches])
    
    prompt = f"""Now let's suppose that the following information is correct:
{match_text}

1. Based on the information provided by the match-up(s) you have found, determine what the MOST LIKELY next pair to match-up is.
2. Express this match-up using the following JSON: {{"%%": "X"}}
where %% is the serial, and X is the corresponding translation.
Do not match-up any other pairs yet."""
    
    return prompt

def extract_json_answer(response_text: str) -> Dict[str, str]:
    """Extract JSON answer from model response"""
    try:
        # Look for JSON in the response - try multiple patterns
        import re
        
        # First try to find a complete JSON object
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{(?:[^{}]|{[^{}]*})*\}',  # Nested JSON object
            r'```json\s*(\{.*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{.*?\})\s*```',  # JSON in generic code blocks
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # If no JSON found, try to extract key-value pairs manually with more flexible patterns
        key_value_patterns = [
            r'"([^"]+)"\s*:\s*"([^"]*)',  # General key-value
            r'"([a-zA-Z0-9.\(\)\s]+)"\s*:\s*"([^"]*)',  # More flexible keys
            r'([a-zA-Z0-9.\(\)\s]+)\s*:\s*([^,}]+)',  # Without quotes
        ]
        
        for pattern in key_value_patterns:
            matches = re.findall(pattern, response_text)
            if matches:
                return {key.strip('"').strip(): value.strip('"').strip() for key, value in matches}
        
        return {}
    except Exception as e:
        print(f"JSON extraction error: {e}")
        return {}

def extract_single_match(response_text: str) -> tuple:
    """Extract a single match pair from model response"""
    try:
        parsed = extract_json_answer(response_text)
        if parsed:
            # Return the first key-value pair
            for key, value in parsed.items():
                return (key, value)
        return ("PARSE_ERROR", "PARSE_ERROR")
    except Exception as e:
        print(f"Match extraction error: {e}")
        return ("PARSE_ERROR", "PARSE_ERROR")

def prompt_model_chained(problem_data: List[Dict], model_details: Dict, dry_run: bool = False) -> List[Dict]:
    """Run iterative matching for Match-up problems"""
    results = []
    
    # Determine problem characteristics
    overall_question_n = problem_data[0]['index'][0]
    
    # Problems that are pure iterative matching
    iterative_only_problems = [74, 167, 179, 185]
    
    # Problems with matching followed by follow-up questions
    matching_followup_problems = {
        127: [("Q 4.1", "initial"), ("Q 4.2", "followup")],
        168: [("Q 3.1", "initial"), ("Q 3.2", "followup")],
        169: [("Q A4.1.", "initial"), ("Q A4.2.", "followup")],
        174: [("Q 4.1", "initial"), ("Q 4.2", "followup")],
        187: [("Q 2.1", "initial"), ("Q 2.2", "followup"), ("Q 2.3", "followup")],
        196: [("Q 6.1", "initial"), ("Q 6.2", "followup"), ("Q 6.3", "followup")],
        220: [("Q 5.1", "initial"), ("Q 5.2", "followup"), ("Q 5.3", "followup"), ("Q 5.4", "followup")]
    }
    
    if overall_question_n in iterative_only_problems:
        # Pure iterative matching
        return process_iterative_matching(problem_data, model_details, dry_run)
    elif overall_question_n in matching_followup_problems:
        # Matching + follow-up questions
        return process_matching_with_followup(problem_data, model_details, dry_run, matching_followup_problems[overall_question_n])
    else:
        print(f"Warning: Unknown problem type for overall_question_n {overall_question_n}")
        return []

def process_iterative_matching(problem_data: List[Dict], model_details: Dict, dry_run: bool) -> List[Dict]:
    """Process problems that are pure iterative matching"""
    results = []
    conversation_history = []
    confirmed_matches = []
    
    # Get the initial question data (should be only one)
    question_data = problem_data[0]
    
    # Determine number of pairs to match from expected answers
    num_pairs = len(question_data['question_details']['subprompts'])
    
    # Step 1: Initial matching prompt
    initial_prompt = create_initial_matching_prompt(problem_data)
    conversation_history.append(initial_prompt)
    
    # Iterative matching process
    for match_round in range(num_pairs):
        if match_round == 0:
            current_prompt = initial_prompt
        else:
            current_prompt = create_next_matching_prompt(confirmed_matches)
            conversation_history.append(current_prompt)
        
        # Create full conversation
        full_conversation = "\n\n".join(conversation_history)
        
        if dry_run:
            raw_response = f'{{"mock_key_{match_round}": "mock_value_{match_round}"}}'
        else:
            # API call for this matching step
            matching_batch = {
                "questions": [full_conversation],
                "answers": [{}]  # Empty for open-ended matching
            }
            
            raw_response = call_model_with_retry(matching_batch, model_details, f"matching round {match_round+1}")
        
        # Extract the match
        serial, translation = extract_single_match(raw_response)
        confirmed_matches.append((serial, translation))
        conversation_history.append(raw_response)
        
        # Create result entry for this match
        result = {
            'overall_question_n': question_data['index'][0],
            'question_n': question_data['question_details']['question_n'],
            'obfuscated_question_n': question_data['index'][1],
            'obf_num': question_data['index'][3],
            'split_key': question_data['split_key'],
            'match_round': match_round + 1,
            'confirmed_matches': confirmed_matches.copy(),
            'model_raw_response': raw_response,
            'model_parsed_response': {serial: translation},
            'expected_answer': {
                subprompt['questionpart_n']: subprompt['answer'] 
                for subprompt in question_data['question_details']['subprompts']
            },
            'question_details': question_data['question_details']
        }
        
        results.append(result)
    
    return results

def process_matching_with_followup(problem_data: List[Dict], model_details: Dict, dry_run: bool, question_structure: List[tuple]) -> List[Dict]:
    """Process problems with initial matching followed by follow-up questions"""
    results = []
    conversation_history = []
    confirmed_matches = []
    
    # Group questions by type
    questions_by_name = {q['question_details']['question_n']: q for q in problem_data}
    
    # Find the initial matching question
    initial_question_name = None
    for q_name, q_type in question_structure:
        if q_type == "initial":
            initial_question_name = q_name
            break
    
    if initial_question_name not in questions_by_name:
        print(f"Error: Initial question {initial_question_name} not found")
        return []
    
    initial_question = questions_by_name[initial_question_name]
    num_pairs = len(initial_question['question_details']['subprompts'])
    
    # Step 1: Iterative matching for initial question
    initial_prompt = create_initial_matching_prompt([initial_question])
    conversation_history.append(initial_prompt)
    
    for match_round in range(num_pairs):
        if match_round == 0:
            current_prompt = initial_prompt
        else:
            current_prompt = create_next_matching_prompt(confirmed_matches)
            conversation_history.append(current_prompt)
        
        full_conversation = "\n\n".join(conversation_history)
        
        if dry_run:
            raw_response = f'{{"mock_key_{match_round}": "mock_value_{match_round}"}}'
        else:
            matching_batch = {
                "questions": [full_conversation],
                "answers": [{}]
            }
            raw_response = call_model_with_retry(matching_batch, model_details, f"matching round {match_round+1}")
        
        serial, translation = extract_single_match(raw_response)
        confirmed_matches.append((serial, translation))
        conversation_history.append(raw_response)
        
        # Create result entry for this match
        result = {
            'overall_question_n': initial_question['index'][0],
            'question_n': initial_question['question_details']['question_n'],
            'obfuscated_question_n': initial_question['index'][1],
            'obf_num': initial_question['index'][3],
            'split_key': initial_question['split_key'],
            'match_round': match_round + 1,
            'confirmed_matches': confirmed_matches.copy(),
            'model_raw_response': raw_response,
            'model_parsed_response': {serial: translation},
            'expected_answer': {
                subprompt['questionpart_n']: subprompt['answer'] 
                for subprompt in initial_question['question_details']['subprompts']
            },
            'question_details': initial_question['question_details']
        }
        
        results.append(result)
    
    # Step 2: Follow-up questions
    for q_name, q_type in question_structure:
        if q_type == "followup" and q_name in questions_by_name:
            followup_question = questions_by_name[q_name]
            
            # Create follow-up prompt
            followup_prompt = create_followup_question_prompt(followup_question)
            conversation_history.append(followup_prompt)
            
            full_conversation = "\n\n".join(conversation_history)
            
            if dry_run:
                raw_response = '{"a": "[DRY RUN] Mock followup answer"}'
            else:
                question_answers = {
                    subprompt['questionpart_n']: '' 
                    for subprompt in followup_question['question_details']['subprompts']
                }
                question_batch = {
                    "questions": [full_conversation],
                    "answers": [question_answers]
                }
                raw_response = call_model_with_retry(question_batch, model_details, f"followup question {q_name}")
            
            parsed_answer = extract_json_answer(raw_response)
            conversation_history.append(raw_response)
            
            # Create result entry for follow-up
            result = {
                'overall_question_n': followup_question['index'][0],
                'question_n': followup_question['question_details']['question_n'],
                'obfuscated_question_n': followup_question['index'][1],
                'obf_num': followup_question['index'][3],
                'split_key': followup_question['split_key'],
                'confirmed_matches': confirmed_matches.copy(),  # Include all matches from initial phase
                'model_raw_response': raw_response,
                'model_parsed_response': parsed_answer,
                'expected_answer': {
                    subprompt['questionpart_n']: subprompt['answer'] 
                    for subprompt in followup_question['question_details']['subprompts']
                },
                'question_details': followup_question['question_details']
            }
            
            results.append(result)
    
    return results

def call_model_with_retry(batch: Dict, model_details: Dict, context: str) -> str:
    """Call model with retry logic"""
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            _, response = prompt_models.prompt_closed_model(batch, model_details, cot=False)
            print(f"{context} response length: {len(response)} chars")
            return response
        except Exception as e:
            print(f"API error for {context} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("Max retries reached, using fallback response")
                return '{"error": "API_FAILED"}'

def load_cache(model_name: str, tmp_path: str = TMP_PATH) -> Dict:
    """Load cached responses"""
    cached = []
    cached_dict = {}
    
    os.makedirs(tmp_path, exist_ok=True)
    
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            if model_name + "_matchup_chained" in file and "tmp" in file:
                with open(Path(root, file), "r", encoding='utf-8') as f:
                    cached.extend(json.load(f))
    
    for entry in cached:
        # Create unique key from identifiers
        key = f"{entry['overall_question_n']}_{entry['question_n']}"
        cached_dict[key] = entry
    
    return cached_dict

def save_cache(responses: List[Dict], model_name: str, tmp_path: str = TMP_PATH):
    """Save responses to cache"""
    os.makedirs(tmp_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_matchup_chained_tmp_{timestamp}.json"
    filepath = Path(tmp_path) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

def main(
    benchmark_file: str = "../data/splits/benchmark_same_obf_matchup.jsonl",
    model_names: str = "DRY_RUN",
    max_problems: int = None,
    use_cache: bool = True
):
    """
    Run chained prompting evaluation on Match-up problems
    
    Args:
        benchmark_file: Path to the filtered Match-up benchmark file
        model_names: Comma-separated list of model names to evaluate
        max_problems: Maximum number of problems to process (for testing)
        use_cache: Whether to use cached responses
    """
    
    # Load model configurations
    with open(MODEL_LIST, 'r') as f:
        model_configs = json.load(f)
    
    # Parse model names
    models_to_evaluate = [name.strip() for name in model_names.split(',')]
    
    # Load questions
    print("Loading questions...")
    questions_by_problem = load_questions(benchmark_file)
    
    if max_problems:
        # Limit to first N problems for testing
        problem_numbers = sorted(list(questions_by_problem.keys()))[:max_problems]
        questions_by_problem = {n: questions_by_problem[n] for n in problem_numbers}
    
    print(f"Loaded {len(questions_by_problem)} problems with Match-up format")
    
    # Process each model
    for model_name in models_to_evaluate:
        if model_name == "DRY_RUN":
            model_details = {"name": "dry_run", "model_type": "dry_run"}
        elif model_name not in model_configs:
            print(f"Warning: Model {model_name} not found in model list")
            continue
        else:
            model_details = model_configs[model_name]
        print(f"\nProcessing model: {model_name}")
        
        # Load cache
        cached_responses = load_cache(model_name) if use_cache else {}
        if use_cache and len(cached_responses) > 0:
            print(f"Loaded {len(cached_responses)} cached responses")
        all_responses = []
        
        # Process each problem
        for problem_n in tqdm(sorted(questions_by_problem.keys()), desc=f"Processing {model_name}"):
            problem_data = questions_by_problem[problem_n]
            
            # Check if all questions in this problem are cached
            all_cached = True
            if use_cache:
                for question_data in problem_data:
                    key = f"{problem_n}_{question_data['question_details']['question_n']}"
                    if key not in cached_responses:
                        all_cached = False
                        break
            else:
                all_cached = False
            
            if all_cached:
                # Use cached responses
                for question_data in problem_data:
                    key = f"{problem_n}_{question_data['question_details']['question_n']}"
                    all_responses.append(cached_responses[key])
            else:
                # Run chained prompting for this problem
                try:
                    problem_responses = prompt_model_chained(problem_data, model_details, dry_run=(model_name == "DRY_RUN"))
                    all_responses.extend(problem_responses)
                    
                    # Save to cache
                    if use_cache:
                        save_cache(problem_responses, model_name)
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error processing problem {problem_n}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Save final results
        os.makedirs(OUTFOLDER, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(OUTFOLDER) / f"{model_name}_matchup_chained_evaluation_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for response in all_responses:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(all_responses)} responses to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)