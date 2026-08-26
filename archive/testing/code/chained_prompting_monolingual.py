#!/usr/bin/env python3
"""
Chained prompting approach for Monolingual problems from benchmark_same_obf dataset.

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
OUTFOLDER = "../data/chained_responses"
TMP_PATH = "../data/chained_responses/tmp"
MODEL_LIST = "../data/model_list.json"

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

def create_reasoning_prompt(problem_data: List[Dict]) -> str:
    """Create the initial reasoning prompt to analyze the language"""
    # Use the first question's context and preamble
    first_question = problem_data[0]
    metadata = first_question['question_details']['metadata']
    
    preamble = metadata.get('preamble', '')
    context = metadata.get('context', '')
    
    prompt = f"""Below is a problem sheet from a linguistics exam. Your task is to determine as much information about the language and its number system as possible, purely from the information provided. You should systematically go through the information provided, and try to determine the vocabulary meaning of each number, the base of the number system (e.g. decimal, hexadecimal), the syntactic structure (such as word order), the morphology, and any other patterns you can see in the language's number system. Test every piece of information you determine against every example provided.

{preamble}
{context}"""
    
    return prompt

def create_question_prompt(question_data: Dict) -> str:
    """Create a prompt for a specific question"""
    prompt = question_data['question_details']['prompt']
    subprompts = question_data['question_details']['subprompts']
    
    # Create the question text
    question_text = f"Based on the information about the language you have determined, solve the following puzzle: \n{prompt}"
    
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
        
        # If no JSON found, try to extract key-value pairs manually
        key_value_pattern = r'"([a-z])"\s*:\s*"([^"]*)"'
        matches = re.findall(key_value_pattern, response_text)
        if matches:
            return {key: value for key, value in matches}
        
        return {}
    except Exception as e:
        print(f"JSON extraction error: {e}")
        return {}

def prompt_model_chained(problem_data: List[Dict], model_details: Dict, dry_run: bool = False) -> List[Dict]:
    """Run chained prompting for a single problem with multiple questions"""
    results = []
    
    # Step 1: Reasoning prompt
    reasoning_prompt = create_reasoning_prompt(problem_data)
    
    if dry_run:
        reasoning_output = "[DRY RUN] This would be the model's reasoning about the language patterns."
    else:
        # Create batch for reasoning
        reasoning_batch = {
            "questions": [reasoning_prompt],
            "answers": [{}]  # Empty dict for reasoning stage
        }
        
        # Get reasoning response with retry logic
        max_retries = 3
        retry_delay = 5  # seconds
        reasoning_output = None
        
        for attempt in range(max_retries):
            try:
                _, reasoning_output = prompt_models.prompt_closed_model(reasoning_batch, model_details, cot=False)
                print(f"Reasoning response length: {len(reasoning_output)} chars")
                break  # Success, exit retry loop
            except Exception as e:
                print(f"API error during reasoning (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Max retries reached, using fallback reasoning")
                    reasoning_output = "[API ERROR] Could not generate reasoning response"
    
    # Step 2: Question prompts in sequence (same context window)
    conversation_history = [reasoning_prompt, reasoning_output]
    
    for question_data in problem_data:
        question_prompt = create_question_prompt(question_data)
        conversation_history.append(question_prompt)
        
        # Create full conversation for this question
        full_conversation = "\n\n".join(conversation_history)
        
        if dry_run:
            raw_response = '{"a": "[DRY RUN] Mock answer"}'
        else:
            # Create batch for this question  
            question_answers = {
                subprompt['questionpart_n']: '' 
                for subprompt in question_data['question_details']['subprompts']
            }
            question_batch = {
                "questions": [full_conversation],
                "answers": [question_answers]
            }
            
            # Get question response with retry logic
            max_retries = 3
            retry_delay = 5  # seconds
            raw_response = None
            
            for attempt in range(max_retries):
                try:
                    _, raw_response = prompt_models.prompt_closed_model(question_batch, model_details, cot=False)
                    print(f"Question response length: {len(raw_response)} chars")
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"API error for question {question_data['question_details']['question_n']} (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("Max retries reached, using fallback response")
                        raw_response = '{"error": "API_FAILED"}'
        
        # Extract JSON answer
        parsed_answer = extract_json_answer(raw_response)
        
        # Add response to conversation history
        conversation_history.append(raw_response)
        
        # Create result entry
        result = {
            'overall_question_n': question_data['index'][0],
            'question_n': question_data['question_details']['question_n'],
            'obfuscated_question_n': question_data['index'][1],
            'obf_num': question_data['index'][3],
            'split_key': question_data['split_key'],
            'reasoning_output': reasoning_output,  # Same for all questions in this problem
            'model_raw_response': raw_response,
            'model_parsed_response': parsed_answer,
            'expected_answer': {
                subprompt['questionpart_n']: subprompt['answer'] 
                for subprompt in question_data['question_details']['subprompts']
            },
            'question_details': question_data['question_details']
        }
        
        results.append(result)
    
    return results

def load_cache(model_name: str, tmp_path: str = TMP_PATH) -> Dict:
    """Load cached responses"""
    cached = []
    cached_dict = {}
    
    os.makedirs(tmp_path, exist_ok=True)
    
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            if model_name + "_chained_monolingual" in file and "tmp" in file:
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
    filename = f"{model_name}_chained_monolingual_tmp_{timestamp}.json"
    filepath = Path(tmp_path) / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

def main(
    benchmark_file: str = "../data/splits/benchmark_same_obf_monolingual.jsonl",
    model_names: str = "DRY_RUN",
    max_problems: int = None,
    use_cache: bool = True
):
    """
    Run chained prompting evaluation on Monolingual problems
    
    Args:
        benchmark_file: Path to the filtered Monolingual benchmark file
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
    
    print(f"Loaded {len(questions_by_problem)} problems with Monolingual format")
    
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
        output_file = Path(OUTFOLDER) / f"{model_name}_chained_monolingual_evaluation_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for response in all_responses:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(all_responses)} responses to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)