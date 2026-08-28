#!/usr/bin/env python3
"""
LLM-as-a-Judge: 'top-1 / multiple-choice judge' (paper §4.2.2, Figure 13).

Ports llm_mc_responses.py from the private research repo. For each
subquestion with more than one unique model answer, presents the full
problem sheet plus the unique answers as multiple-choice options and asks
the judge LLM to pick the single best one.

Input is a CSV produced by evaluate_baseline.py or subsample_eval.py.
"""

import ast
import csv
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fire
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarking"))
import prompt_models

csv.field_size_limit(sys.maxsize)

MAX_API_ATTEMPT = 10
OUTFOLDER = "../../data/mc_judge_output"
TMP_PATH = "../../data/mc_judge_output/tmp"
MODEL_LIST = "../../data/model_list.json"


def load_cache(model_name, dataset_name=None, tmp_path=TMP_PATH, suffix="_mc"):
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


def get_answer_frequencies(model_answers: List[str]) -> Dict[str, int]:
    valid = [a for a in model_answers if a and a != "N/A" and not (isinstance(a, str) and a.startswith("IMPROPER PARSING:"))]
    return dict(Counter(valid))


def get_top_frequent_answers(answer_frequencies: Dict[str, int], max_options: int) -> List[str]:
    sorted_answers = sorted(answer_frequencies.items(), key=lambda x: (-x[1], x[0]))
    return [a for a, _ in sorted_answers[:max_options]]


def clean_problem_sheet(problem_sheet: str) -> str:
    duplicate_instruction = (
        "Below is a problem sheet from a lingusitics exam. You will first see the entire sheet, "
        "then be asked to respond to specific questions from the sheet. Your answers to the "
        "questions should rely only on reasoning about the information provided in the sheet."
    )
    if problem_sheet.startswith(duplicate_instruction):
        problem_sheet = problem_sheet[len(duplicate_instruction):].strip()

    problem_sheet = re.sub(r"Now respond to the following questions:.*$", "", problem_sheet,
                            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    for pattern in [
        r"Make sure to finish your answer with json output with the keys as provided below:\s*\{[^}]*\}\s*",
        r"Make sure to finish your answer with json output[^{]*\{[^}]*\}\s*",
        r'\{[^}]*"a\."[^}]*\}\s*$',
    ]:
        problem_sheet = re.sub(pattern, "", problem_sheet, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return problem_sheet.strip()


def extract_subquestion_text(problem_sheet: str, question_n: str, serial: str) -> str:
    cleaned_sheet = clean_problem_sheet(problem_sheet)

    now_respond_match = re.search(
        r"Now respond to the following questions:(.*?)(?=Make sure to finish|$)",
        problem_sheet, re.DOTALL | re.IGNORECASE,
    )
    question_content = None
    if now_respond_match:
        question_content = now_respond_match.group(1)
    else:
        question_section_pattern = rf"{re.escape(question_n)}[^Q]*?(?=Q\s+\d|\Z)"
        m = re.search(question_section_pattern, cleaned_sheet, re.DOTALL | re.IGNORECASE)
        if m:
            question_content = m.group(0)

    if question_content:
        lines = question_content.split("\n")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith(serial + " "):
                content = line_stripped[len(serial):].strip()
                if content and len(content) > 3 and not re.match(r"^[A-Z]\s*$", content):
                    return f"{serial} {content}"
                if content and len(content) <= 3:
                    for j in range(max(0, i - 15), i):
                        prev_line = lines[j].strip()
                        if ("?" in prev_line or "give the answer" in prev_line.lower()
                                or "how would" in prev_line.lower() or "how is it pronounced" in prev_line.lower()):
                            return f"{serial} {prev_line}"
            elif line_stripped == serial:
                context_found = None
                for j in range(max(0, i - 15), i):
                    prev_line = lines[j].strip()
                    if "which two words are they" in prev_line.lower():
                        context_found = prev_line
                    elif "?" in prev_line and any(w in prev_line.lower() for w in ["how", "what", "give", "pronounced"]):
                        context_found = context_found or prev_line
                if context_found:
                    return f"{serial} {context_found}"

    if re.search(r"Which two words are they\?", cleaned_sheet, re.IGNORECASE):
        pos = cleaned_sheet.lower().find("which two words are they?")
        if pos > -1 and re.search(rf"\b{re.escape(serial)}\b", cleaned_sheet[pos:pos + 200]):
            return f"{serial} Which two words are they?"

    for pattern in [
        rf"Translate into Language X:[^{serial}]*{re.escape(serial)}\s+([^\n]+)",
        rf"Translate.*Language X.*{re.escape(serial)}\s+([^\n]+)",
    ]:
        m = re.search(pattern, cleaned_sheet, re.IGNORECASE | re.DOTALL)
        if m and m.group(1).strip():
            return f"{serial} {m.group(1).strip()}"

    for pattern in [
        rf"how would this complex word be pronounced.*{re.escape(serial)}\s+([A-Z])",
        rf"how is it pronounced.*{re.escape(serial)}\s+([A-Z])",
        rf"how would you say.*{re.escape(serial)}\s+([A-Z])",
    ]:
        m = re.search(pattern, cleaned_sheet, re.IGNORECASE | re.DOTALL)
        if m:
            return f"{serial} Dialect {m.group(1)}"

    lines = cleaned_sheet.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == serial:
            for j in range(max(0, i - 20), i):
                prev_line = lines[j].strip()
                if "?" in prev_line and any(w in prev_line.lower() for w in ["how", "what", "which", "give", "pronounced", "differ"]):
                    return f"{serial} {prev_line}"

    return f"Answer question {serial}."


def create_mc_judge_prompt(full_problem_sheet: str, question_n: str, serial: str, options: List[str]) -> str:
    cleaned_problem_sheet = clean_problem_sheet(full_problem_sheet)
    subquestion_text = extract_subquestion_text(full_problem_sheet, question_n, serial)

    parts = subquestion_text.split(" ", 1)
    extracted_serial = parts[0]
    serial_content = ""
    question_text = f"Answer question {extracted_serial}:"

    if len(parts) > 1:
        content = parts[1]
        if "?" in content:
            question_text = content
        else:
            serial_content = content
            now_respond_match = re.search(
                r"Now respond to the following questions:(.*?)(?=Make sure to finish|$)",
                full_problem_sheet, re.DOTALL | re.IGNORECASE,
            )
            if now_respond_match:
                lines = now_respond_match.group(1).split("\n")
                serial_line_idx = None
                for i, line in enumerate(lines):
                    if line.strip().startswith(extracted_serial + " ") or line.strip() == extracted_serial:
                        serial_line_idx = i
                        break
                if serial_line_idx is not None:
                    for j in range(serial_line_idx - 1, max(0, serial_line_idx - 20), -1):
                        prev_line = lines[j].strip()
                        if "translate" in prev_line.lower() and ":" in prev_line and len(prev_line) > 10:
                            question_text = prev_line.rstrip(":")
                            break
                        if prev_line.endswith(":") and any(w in prev_line.lower() for w in ["translate", "what", "how", "which", "give", "complete"]):
                            question_text = prev_line.rstrip(":")
                            break

    sorted_options = sorted(options)
    options_text = "\n".join(sorted_options)

    prompt_parts = [
        "Below is a problem sheet from a linguistics exam. You will first see the entire sheet, then be "
        "asked to respond to a specific subquestion from the sheet. You will be given a set of options to "
        "choose from. Your answers to the questions should rely only on reasoning about the information "
        "provided in the sheet.",
        cleaned_problem_sheet,
        "Now provide the answer to the following subquestion.",
        question_text,
        f"{extracted_serial} {serial_content}" if serial_content else extracted_serial,
        "These are the options you have to choose from:",
        options_text,
        "Consider the logic that could be used to lead to each of the options presented. Based on your "
        "reasoning, please select one option only that you think is the correct answer. If you think "
        "multiple options could be correct, select only one of them. Your output MUST end with a valid "
        'JSON dictionary in this exact format: {"answer": "option"} (where \'option\' is the EXACT text '
        "of the option you have selected).",
    ]
    return "\n\n".join(prompt_parts)


def parse_mc_judge_response(response: str, options: List[str]) -> Dict[str, Any]:
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            return {"error": "No JSON found in response", "valid": False}
        result = json.loads(json_match.group(0))
        if "answer" not in result:
            return {"error": "No 'answer' key found in JSON response", "valid": False}
        selected_answer = result["answer"]
        if selected_answer in options:
            return {"selected_answer": selected_answer, "valid": True}
        return {"error": f"Selected answer '{selected_answer}' not in options list", "selected_answer": selected_answer, "valid": False}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "valid": False}
    except Exception as e:
        return {"error": f"Parse error: {e}", "valid": False}


def extract_dataset_name_from_path(csv_path: str) -> str:
    filename = Path(csv_path).stem
    dataset_name = filename.replace("subquestion_eval_", "").replace("_eval", "").replace("evaluation_", "")
    return dataset_name.replace("_", "-")[:50]


def load_evaluation_csv(csv_path, question_filter=None, serial_filter=None, limit=None,
                         only_any_correct=False, min_unique_answers=2, min_correct=None,
                         max_correct=None, format_filter=None):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
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

            model_answers = []
            v = 1
            while f"model_answer_v{v}" in row:
                model_answers.append(row[f"model_answer_v{v}"])
                v += 1

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
            row["answer_frequencies"] = get_answer_frequencies(model_answers)
            row["model_answers"] = model_answers
            rows.append(row)
            if limit and len(rows) >= limit:
                break

    print(f"Loaded {len(rows)} rows for evaluation")
    return rows


def mc_judge_pipeline(
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
    max_options: int = None,
    min_correct: int = None,
    max_correct: int = None,
    format_filter: str = None,
):
    print(f"Starting LLM Multiple Choice Judge evaluation with model: {model}")

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
    output_filename = "_".join([model_name.split("/")[-1], dataset_name, "mc_judge_evaluation", timestamp]) + ".jsonl"

    cached_dict = load_cache(model_name.split("/")[-1], dataset_name=dataset_name) if use_cache else {}
    if use_cache:
        print(f"Found {len(cached_dict)} cached responses")

    Path(outfolder).mkdir(parents=True, exist_ok=True)
    Path(TMP_PATH).mkdir(parents=True, exist_ok=True)

    results, errors = [], []
    for i, row in enumerate(tqdm(rows, desc="MC judging responses")):
        row_id = f"{row['overall_question_n']}_{row['question_n']}_{row['serial']}"

        if use_cache and row_id in cached_dict:
            results.append(cached_dict[row_id])
            continue

        options = row["unique_answers_parsed"]
        if max_options and len(options) > max_options:
            options = get_top_frequent_answers(row["answer_frequencies"], max_options)
        if not options:
            continue

        full_problem_sheet = row["questions"]
        mc_judge_prompt = create_mc_judge_prompt(full_problem_sheet, row["question_n"], row["serial"], options)

        batch = {
            "questions": [mc_judge_prompt],
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
                parse_result = parse_mc_judge_response(raw_output, options)
                subquestion_text = extract_subquestion_text(full_problem_sheet, row["question_n"], row["serial"])

                results.append({
                    "overall_question_n": int(row["overall_question_n"]),
                    "question_n": row["question_n"],
                    "serial": row["serial"],
                    "row_id": row_id,
                    "subquestion": subquestion_text,
                    "options": options,
                    "unique_answers": row["unique_answers_parsed"],
                    "correct_answer": row["correct_answer"],
                    "mc_judge_prompt": mc_judge_prompt,
                    "mc_judge_raw_response": raw_output,
                    "parse_result": parse_result,
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "max_options": max_options,
                })
                success = True
            except Exception as e:
                print(f"Error for {row_id}, attempt {attempt + 1}: {e}")
                traceback.print_exc()
                attempt += 1
                if attempt < MAX_API_ATTEMPT:
                    time.sleep(min(10 * attempt, 60))
                else:
                    errors.append({"row_id": row_id, "error": str(e), "options": options})

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

    print(f"MC Judge evaluation complete! Processed {len(results)} rows with {len(errors)} errors.")


if __name__ == "__main__":
    fire.Fire(mc_judge_pipeline)
