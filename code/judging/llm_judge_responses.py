#!/usr/bin/env python3
"""
LLM-as-a-Judge: 'reranking judge' (paper §4.2.2, Figure 12).

Ports llm_judge_responses.py from the private research repo. For each
subquestion with more than one unique model answer, asks a judge LLM to rank
all unique answers from most to least likely correct; the answer ranked 1 is
taken as the judge's pick.

Input is a CSV produced by evaluate_baseline.py or subsample_eval.py (must
have 'questions', 'serial', and 'unique_answers' columns).
"""

import ast
import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fire
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarking"))
import prompt_models

MAX_API_ATTEMPT = 10
OUTFOLDER = "../../data/judge_output"
TMP_PATH = "../../data/judge_output/tmp"
MODEL_LIST = "../../data/model_list.json"


def load_cache(model_name, dataset_name=None, tmp_path=TMP_PATH, suffix="_judge"):
    cached = []
    cached_dict = {}
    for root, _dirs, files in os.walk(tmp_path):
        for file in files:
            expected_pattern = model_name
            if dataset_name:
                expected_pattern += "_" + dataset_name
            expected_pattern += suffix
            if expected_pattern in file and "tmp" in file:
                with open(Path(root, file), encoding="utf-8") as f:
                    cached.extend(json.load(f))

    for entry in cached:
        row_id = f"{entry['overall_question_n']}_{entry['question_n']}_{entry['serial']}"
        cached_dict[row_id] = entry
    return cached_dict


def parse_subquestion_from_text(questions_text: str, serial: str) -> str:
    preamble_pattern = r"Below is a problem sheet.*?Your answers.*?sheet\."
    text_without_preamble = re.sub(preamble_pattern, "", questions_text, flags=re.DOTALL).strip()

    main_question_match = re.search(
        r"(Question \d+:.*?)(?=\s*Now respond to the following questions:|$)",
        text_without_preamble, re.DOTALL,
    )
    if main_question_match:
        main_question = main_question_match.group(1).strip()
    else:
        main_question = re.split(r"Now respond to the following questions:", text_without_preamble)[0].strip()

    now_respond_match = re.search(
        r"Now respond to the following questions:\s*(.*?)(?=Make sure to finish|$)",
        questions_text, re.DOTALL,
    )
    subquestion_content = f"Answer question {serial}"
    if now_respond_match:
        respond_section = now_respond_match.group(1).strip()
        for line in respond_section.split("\n"):
            line = line.strip()
            if not line:
                continue
            if re.match(rf"^{re.escape(serial)}\s", line):
                subquestion_content = line
                break
            if f"{serial} " in line or line.endswith(f"{serial}"):
                if line.endswith(f"{serial}"):
                    question_part = re.sub(rf"\s*{serial}\s*$", "", line).strip()
                    subquestion_content = f"{question_part}\n{serial}"
                else:
                    subquestion_content = line
                break

    return f"{main_question}\n\n{subquestion_content}".strip()


def create_judge_prompt(puzzle_text: str, unique_answers: List[str]) -> str:
    answers_section = "\n".join(f"{i}. {a}" for i, a in enumerate(unique_answers, 1))
    json_template_lines = [f'  "answer_{i}": 0.0' + ("," if i < len(unique_answers) else "") for i in range(1, len(unique_answers) + 1)]
    json_template = "{\n" + "\n".join(json_template_lines) + "\n}"

    return f"""Evaluate the following solutions to a linguistic puzzle and return ONLY a JSON dictionary with scores.

PUZZLE: {puzzle_text}

POSSIBLE ANSWERS:
{answers_section}

Evaluate each answer's correctness and assign a ranking to each one, where 1 is the answer that you think is most likely to be correct, 2 is the next most likely, etc.

Consider:
- Whether each answer could satisfy the puzzle constraints
- The possible logical reasoning behind each answer
- How well it addresses what the puzzle is asking

Output ONLY a valid JSON dictionary in this exact format:
{json_template}

Use the exact answer numbers as keys. Include ALL answers. Output no other text."""


def parse_judge_response(response: str, num_answers: int) -> Dict[str, float]:
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            return {"error": "No JSON found in response"}
        scores = json.loads(json_match.group(0))
        expected_keys = {f"answer_{i}" for i in range(1, num_answers + 1)}
        if set(scores.keys()) != expected_keys:
            return {"error": f"Missing or extra keys in response: got {set(scores.keys())}"}
        return {k: float(v) for k, v in scores.items()}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}"}
    except Exception as e:
        return {"error": f"Parse error: {e}"}


def extract_dataset_name_from_path(csv_path: str) -> str:
    filename = Path(csv_path).stem
    dataset_name = filename.replace("subquestion_eval_", "").replace("_eval", "").replace("evaluation_", "")
    return dataset_name.replace("_", "-")[:50]


def load_evaluation_csv(csv_path, question_filter=None, serial_filter=None, limit=None,
                         only_any_correct=False, min_unique_answers=1,
                         min_correct=None, max_correct=None, format_filter=None):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        csv.field_size_limit(sys.maxsize)
        for i, row in enumerate(csv.DictReader(f)):
            if question_filter and int(row["overall_question_n"]) != question_filter:
                continue
            if serial_filter and row["serial"] != serial_filter:
                continue
            if only_any_correct and row["is_any_correct"].lower() != "true":
                continue
            if min_correct is not None or max_correct is not None:
                try:
                    number_correct = int(row["number_correct"])
                except (ValueError, KeyError):
                    continue
                if min_correct is not None and number_correct <= min_correct:
                    continue
                if max_correct is not None and number_correct >= max_correct:
                    continue
            if format_filter is not None and row.get("format") != format_filter:
                continue

            try:
                unique_answers = ast.literal_eval(row["unique_answers"])
                if not isinstance(unique_answers, list):
                    continue
                unique_answers = [
                    a for a in unique_answers
                    if a != "N/A" and not (isinstance(a, str) and a.startswith("IMPROPER PARSING:"))
                ]
                if len(unique_answers) < min_unique_answers:
                    continue
            except Exception as e:
                print(f"Warning: could not parse unique_answers for row {i}: {e}")
                continue

            row["unique_answers_parsed"] = unique_answers
            rows.append(row)
            if limit and len(rows) >= limit:
                break

    print(f"Loaded {len(rows)} rows for evaluation")
    return rows


def judge_pipeline(
    model: str,
    response_data_csv: str,
    model_list_path: str = MODEL_LIST,
    outfolder: str = OUTFOLDER,
    use_cache: bool = True,
    question_filter: int = None,
    serial_filter: str = None,
    limit: int = None,
    only_any_correct: bool = False,
    min_unique_answers: int = 2,
    min_correct: int = None,
    max_correct: int = None,
    format_filter: str = None,
):
    print(f"Starting LLM Judge evaluation with model: {model}")

    with open(model_list_path) as f:
        model_list = json.load(f)
    if model not in model_list:
        raise ValueError(f"Model {model} not found in model list")
    model_details = model_list[model]
    model_name = model_details["name"]

    rows = load_evaluation_csv(
        response_data_csv, question_filter=question_filter, serial_filter=serial_filter,
        limit=limit, only_any_correct=only_any_correct, min_unique_answers=min_unique_answers,
        min_correct=min_correct, max_correct=max_correct, format_filter=format_filter,
    )
    if not rows:
        print("No rows to process (nothing has >= min_unique_answers unique answers). Exiting.")
        return

    dataset_name = extract_dataset_name_from_path(response_data_csv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = "_".join([model_name.split("/")[-1], dataset_name, "judge_evaluation", timestamp]) + ".jsonl"

    cached_dict = load_cache(model_name.split("/")[-1], dataset_name=dataset_name) if use_cache else {}
    if use_cache:
        print(f"Found {len(cached_dict)} cached responses")

    Path(outfolder).mkdir(parents=True, exist_ok=True)
    Path(TMP_PATH).mkdir(parents=True, exist_ok=True)

    results, errors = [], []
    for i, row in enumerate(tqdm(rows, desc="Judging responses")):
        row_id = f"{row['overall_question_n']}_{row['question_n']}_{row['serial']}"

        if use_cache and row_id in cached_dict:
            results.append(cached_dict[row_id])
            continue

        puzzle_text = parse_subquestion_from_text(row["questions"], row["serial"])
        unique_answers = row["unique_answers_parsed"]
        judge_prompt = create_judge_prompt(puzzle_text, unique_answers)

        batch = {
            "questions": [judge_prompt],
            "answers": [{}],
            "index": [[row["overall_question_n"], row["question_n"], row["serial"]]],
            "metadata": [{"row_id": row_id}],
        }

        attempt, success = 0, False
        while attempt < MAX_API_ATTEMPT and not success:
            try:
                if model_details["model_type"] not in ["openai", "anthropic", "cohere", "google", "open_router"]:
                    raise ValueError(f"Unsupported model type for judging: {model_details['model_type']}")
                _responses, raw_output = prompt_models.prompt_closed_model(batch, model_details, cot=False)
                scores = parse_judge_response(raw_output, len(unique_answers))

                results.append({
                    "overall_question_n": int(row["overall_question_n"]),
                    "question_n": row["question_n"],
                    "serial": row["serial"],
                    "row_id": row_id,
                    "puzzle_text": puzzle_text,
                    "unique_answers": unique_answers,
                    "judge_prompt": judge_prompt,
                    "judge_raw_response": raw_output,
                    "scores": scores,
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                })
                success = True
            except Exception as e:
                print(f"Error for {row_id}, attempt {attempt + 1}: {e}")
                traceback.print_exc()
                attempt += 1
                if attempt < MAX_API_ATTEMPT:
                    time.sleep(min(10 * attempt, 60))
                else:
                    errors.append({"row_id": row_id, "error": str(e), "unique_answers": unique_answers})

        if (i + 1) % 5 == 0:
            tmp_path = Path(TMP_PATH) / f"{output_filename.replace('.jsonl', '')}_tmp_{i + 1}.json"
            try:
                tmp_path.write_text(json.dumps(results[-5:], indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Warning: could not save temporary results: {e}")

    output_path = Path(outfolder) / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {output_path}")

    if errors:
        error_path = Path(outfolder) / f"{output_filename.replace('.jsonl', '')}_errors.json"
        error_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False))
        print(f"Saved {len(errors)} errors to {error_path}")

    print(f"Judge evaluation complete! Processed {len(results)} rows with {len(errors)} errors.")


if __name__ == "__main__":
    fire.Fire(judge_pipeline)
