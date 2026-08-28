#!/usr/bin/env python3
"""
Evaluate multiple runs of Match-up (Tree-of-Matches) chained prompting at
the subquestion level -- port of the private research repo's
evaluate_matchup_multi_run.py.

Match-up answers are pairings extracted from chained_prompting_matchup.py's
'confirmed_matches' rounds, not simple string answers, so this doesn't share
scoring logic with eval_common.py / the rest of code/judging/ -- it's a
self-contained port, same as the original.

Unlike the other chained-prompting formats, this script needs no separate
"collect versions" helper: chained_prompting_matchup.py writes
'<model_name>_matchup_chained_evaluation_<timestamp>.jsonl' files, and this
script auto-discovers and auto-versions them itself. You just need to have
actually run chained_prompting_matchup.py num_versions times for the model
first (see code/chained_prompting/chained_prompting_matchup.py).

What changed from the private repo's version:
- get_model_file_pattern() hardcoded a lookup from 3 fixed model aliases
  ('gemini', 'deepseek', 'llama') to specific model_list.json checkpoint
  names. That mapping is gone -- 'model_name' is now the exact string
  chained_prompting_matchup.py used (i.e. the key in data/model_list.json),
  since that's what actually appears in the output filenames.
- All paths are relative to this repo (data/chained_responses/,
  data/matchup_analysis/) instead of testing/data/... and
  openrouter_runs/...

Usage:
    python evaluate_matchup_multi_run.py --model_name Sonnet --num_versions 32
"""

import csv
import os
import random
import re
import shutil
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import fire

from eval_common import is_valid_model_answer, normalize_answer

import json


def calculate_majority_and_tiebreaker(model_answers):
    """Same as eval_common's version, but returns unique_answers too (this
    script's CSV records unique_answers as a plain list, not a JSON string
    of only valid non-duplicate answers filtered a second time).

    Does NOT seed the RNG here -- see eval_common.calculate_majority_and_tiebreaker
    for why: this is called once per subquestion, and reseeding to a fixed
    value before every call would make every 2-way tie resolve to the same
    list position every time instead of varying by row. main() seeds once
    for whole-run reproducibility instead."""
    valid_answers = [a for a in model_answers if is_valid_model_answer(a)]
    if not valid_answers:
        return "", "", 0, []

    answer_counts = Counter(valid_answers)
    unique_answers = list(answer_counts.keys())
    max_count = max(answer_counts.values())
    most_common = [a for a, c in answer_counts.items() if c == max_count]

    if len(most_common) == 1:
        majority = tiebreaker = most_common[0]
    else:
        majority = "N/A"
        tiebreaker = random.choice(most_common)

    return majority, tiebreaker, max_count, unique_answers


def find_model_files(input_dir: str, model_name: str) -> List[str]:
    """Find all matchup chained-prompting JSONL outputs for this model, sorted by timestamp."""
    files = []
    for filename in os.listdir(input_dir):
        if filename.startswith(model_name) and "_matchup_chained_evaluation_" in filename and filename.endswith(".jsonl"):
            files.append(os.path.join(input_dir, filename))

    def extract_timestamp(filepath):
        match = re.search(r"(\d{8}_\d{6})", os.path.basename(filepath))
        return match.group(1) if match else os.path.basename(filepath)

    files.sort(key=extract_timestamp)
    return files


def copy_and_version_files(source_files: List[str], output_dir: str, num_versions: int) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    if len(source_files) < num_versions:
        raise ValueError(f"Requested {num_versions} versions but only {len(source_files)} files found")

    versioned_files = []
    for i in range(num_versions):
        basename = os.path.basename(source_files[i])
        dest_path = os.path.join(output_dir, f"v{i + 1}_{basename}")
        shutil.copy2(source_files[i], dest_path)
        versioned_files.append(dest_path)
        print(f"Copied {basename} -> v{i + 1}_{basename}")

    return versioned_files


def parse_matchup_file(file_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: failed to parse line {line_num} in {file_path}: {e}")
    return results


def is_matchup_question(entry: Dict[str, Any]) -> bool:
    return "match_round" in entry and entry.get("match_round") is not None


def clean_overall_question_n(question_id: str) -> str:
    return question_id.split("_")[0] if "_" in question_id else question_id


def normalize_part_id(part_id):
    if not isinstance(part_id, str):
        part_id = str(part_id)
    return part_id.lower().rstrip(".").rstrip(":").strip("()[]{}")


def extract_answers_from_file(file_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """overall_question_id -> sub_question_id -> {serial: answer}"""
    question_data = parse_matchup_file(file_path)
    if not question_data:
        return {}

    questions = defaultdict(lambda: defaultdict(list))
    for entry in question_data:
        overall_question_id = entry.get("obfuscated_question_n", entry.get("overall_question_n", "unknown"))
        sub_question_id = entry.get("question_n", "main")
        questions[overall_question_id][sub_question_id].append(entry)

    final_answers = {}
    for overall_question_id, sub_questions in questions.items():
        final_answers[overall_question_id] = {}

        for sub_question_id, entries in sub_questions.items():
            if not entries:
                continue
            entries.sort(key=lambda x: x.get("match_round", 0))

            if is_matchup_question(entries[0]):
                final_entry = entries[-1]  # highest match_round
                final_answer = {
                    pair[0]: pair[1]
                    for pair in final_entry.get("confirmed_matches", [])
                    if len(pair) >= 2
                }
                final_answers[overall_question_id][sub_question_id] = final_answer
            else:
                question_details = entries[0].get("question_details", {})
                if "subprompts" in question_details:
                    subprompts = question_details.get("subprompts", [])
                else:
                    questions_list = question_details.get("questions", [])
                    if isinstance(questions_list, str):
                        try:
                            questions_list = json.loads(questions_list)
                        except Exception:
                            questions_list = []
                    subprompts = next(
                        (q.get("subprompts", []) for q in questions_list if q.get("question_n") == sub_question_id),
                        [],
                    )

                expected_parts = [sp.get("questionpart_n", "") for sp in subprompts]
                final_answer = {part: "" for part in expected_parts}
                normalized_to_expected = {normalize_part_id(p): p for p in expected_parts}

                seen_matches = set()
                for entry in entries:
                    for question_part, answer in entry.get("model_parsed_response", {}).items():
                        expected_part = normalized_to_expected.get(normalize_part_id(question_part))
                        if expected_part and expected_part not in seen_matches:
                            final_answer[expected_part] = answer
                            seen_matches.add(expected_part)

                final_answers[overall_question_id][sub_question_id] = final_answer

    return final_answers


def get_question_context(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    metadata = entries[0].get("question_details", {}).get("metadata", {})
    parts = [metadata.get("preamble", "").strip(), metadata.get("context", "").strip()]
    return "\n\n".join(p for p in parts if p)


def get_expected_answers(entries: List[Dict[str, Any]]) -> Dict[str, str]:
    return entries[0].get("expected_answer", {}) if entries else {}


def main(model_name: str, num_versions: int, input_dir: str = "../../data/chained_responses",
         runs_dir: str = None, output_dir: str = "../../data/matchup_analysis"):
    """
    model_name: the exact model_list.json key used when chained_prompting_matchup.py
        was run (this is what appears in its output filenames).
    num_versions: how many of the available runs to use (e.g. 32, 16, 8, ...).
    input_dir: where chained_prompting_matchup.py's raw output files live.
    runs_dir: where to copy/version the selected files; defaults to
        '<output_dir>/runs/<model_name>_chained_matchup'.
    output_dir: where to write the final CSV.
    """
    random.seed(42)  # one-time, for whole-run reproducibility -- see calculate_majority_and_tiebreaker

    if runs_dir is None:
        runs_dir = str(Path(output_dir) / "runs" / f"{model_name}_chained_matchup")

    print(f"Looking for {model_name} files in {input_dir}...")
    model_files = find_model_files(input_dir, model_name)
    if not model_files:
        raise FileNotFoundError(f"No matchup chained-prompting files found for '{model_name}' in {input_dir}")
    print(f"Found {len(model_files)} files for {model_name}")

    if num_versions > len(model_files):
        raise ValueError(f"Requested {num_versions} versions but only {len(model_files)} files available")

    versioned_files = copy_and_version_files(model_files, runs_dir, num_versions)

    print(f"\nExtracting answers from {num_versions} versions...")
    all_version_answers = []
    all_questions_data = {}

    for i, file_path in enumerate(versioned_files):
        print(f"Processing version {i + 1}...")
        all_version_answers.append(extract_answers_from_file(file_path))

        if i == 0:
            question_data = parse_matchup_file(file_path)
            questions = defaultdict(lambda: defaultdict(list))
            for entry in question_data:
                overall_question_id = entry.get("obfuscated_question_n", entry.get("overall_question_n", "unknown"))
                sub_question_id = entry.get("question_n", "main")
                questions[overall_question_id][sub_question_id].append(entry)

            for overall_question_id, sub_questions in questions.items():
                all_questions_data[overall_question_id] = {
                    sub_question_id: {
                        "context": get_question_context(entries),
                        "expected_answers": get_expected_answers(entries),
                    }
                    for sub_question_id, entries in sub_questions.items()
                }

    print("\nPreparing CSV data...")
    csv_rows = []
    for overall_question_id, sub_questions in all_questions_data.items():
        clean_question_id = clean_overall_question_n(overall_question_id)

        for sub_question_id, question_info in sub_questions.items():
            context = question_info["context"]
            expected_answers = question_info["expected_answers"]

            for serial, correct_answer in expected_answers.items():
                model_answers = []
                for version_answers in all_version_answers:
                    answer = ""
                    sub_answers = version_answers.get(overall_question_id, {}).get(sub_question_id, {})
                    answer = sub_answers.get(serial, "")
                    if not answer:
                        normalized_serial = serial.rstrip(".").rstrip(":").strip("()[]{}")
                        answer = sub_answers.get(normalized_serial, "")
                        if not answer and not any(c in serial for c in ".:()"):
                            answer = sub_answers.get(serial + ".", "")
                    model_answers.append(answer)

                majority, tiebreaker, majority_size, unique_answers = calculate_majority_and_tiebreaker(model_answers)
                number_correct = sum(
                    1 for a in model_answers if is_valid_model_answer(a) and normalize_answer(a) == normalize_answer(correct_answer)
                )
                is_any_correct = number_correct > 0
                is_majority_correct = normalize_answer(majority) == normalize_answer(correct_answer) if is_valid_model_answer(majority) else ""
                is_tiebreaker_correct = normalize_answer(tiebreaker) == normalize_answer(correct_answer) if is_valid_model_answer(tiebreaker) else ""

                row = {
                    "questions": context,
                    "overall_question_n": clean_question_id,
                    "question_n": sub_question_id,
                    "serial": serial,
                    "format": "Match-up",
                    "correct_answer": correct_answer,
                    "majority": majority,
                    "tiebreaker": tiebreaker,
                    "majority_size": majority_size,
                    "unique_answers": str(unique_answers),
                    "is_majority_correct": is_majority_correct,
                    "is_tiebreaker_correct": is_tiebreaker_correct,
                    "is_any_correct": is_any_correct,
                    "number_correct": number_correct,
                }
                for i, answer in enumerate(model_answers, start=1):
                    row[f"model_answer_v{i}"] = answer
                    row[f"is_v{i}_correct"] = normalize_answer(answer) == normalize_answer(correct_answer) if is_valid_model_answer(answer) else ""

                csv_rows.append(row)

    if not csv_rows:
        print("No data to write to CSV")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"{model_name}_matchup_v{num_versions}_evaluation_{timestamp}.csv"

    base_columns = ["questions", "overall_question_n", "question_n", "serial", "format", "correct_answer"]
    version_columns = [c for i in range(1, num_versions + 1) for c in (f"model_answer_v{i}", f"is_v{i}_correct")]
    final_columns = [
        "majority", "tiebreaker", "majority_size", "unique_answers",
        "is_majority_correct", "is_tiebreaker_correct", "is_any_correct", "number_correct",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_columns + version_columns + final_columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV written to: {csv_path}")
    print(f"Processed {len(csv_rows)} subquestions across {len(all_questions_data)} questions")


if __name__ == "__main__":
    fire.Fire(main)
