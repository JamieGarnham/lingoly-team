#!/usr/bin/env python3
"""
Evaluate chained-prompting (Rosetta / Pattern / Monolingual) responses at
the subquestion level, across repeated runs, combining all three formats
into one CSV -- port of the private research repo's
evaluate_chained_subquestions.py.

Reads openrouter_runs/<model_name>_chained_<format>/v{N}_*evaluation*.jsonl
for each format (see collect_chained_versions.py to produce that layout
from chained_prompting_{monolingual,pattern,rosetta}.py's raw output), and
scores each subquestion's model_parsed_response against expected_answer
using the same majority-vote / correctness logic as the rest of
code/judging/ (eval_common.py).

What changed from the private repo's version:
- Format lookup uses data/overall_question_format_mapping.csv (same file
  create_benchmark_same_obf.py uses) instead of testing/data/past-exam-papers.csv.
- The private repo's separate 'format_fixed' column (from a
  question_formats_fixed.csv correction file that doesn't exist here) is
  dropped -- only 'format' is emitted, consistent with the rest of
  code/judging/.
- Uses eval_common.py for correctness/majority-vote logic instead of a
  third copy of the same functions.

Usage:
    python evaluate_chained_subquestions.py Sonnet 32
    # reads openrouter_runs/Sonnet_chained_{monolingual,pattern,rosetta}/v*.jsonl
    # writes openrouter_analysis/sonnet/chained_subquestion_eval_sonnet_all_formats_32.csv
"""

import json
import random
import re
from pathlib import Path

import fire

from eval_common import (
    calculate_majority_and_tiebreaker,
    check_answer_correctness,
    parse_correct_answer,
    unique_valid_answers,
)

FORMAT_MAPPING_CSV = "../../data/overall_question_format_mapping.csv"
FORMATS = ["monolingual", "pattern", "rosetta"]


def load_format_mapping(path=FORMAT_MAPPING_CSV):
    import csv
    mapping = {}
    if not Path(path).exists():
        print(f"Warning: {path} not found. 'format' column will be 'Unknown'.")
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[int(row["overall_question_n"])] = row["format"]
    return mapping


def load_chained_prompt_responses(chained_dir, sample_size=None):
    chained_dir = Path(chained_dir)
    if not chained_dir.exists():
        raise FileNotFoundError(
            f"{chained_dir} does not exist. Run collect_chained_versions.py first."
        )

    model_files = []
    for file_path in chained_dir.glob("v*_*evaluation*.jsonl"):
        match = re.match(r"v(\d+)_.*evaluation.*\.jsonl", file_path.name)
        if match:
            model_files.append((int(match.group(1)), file_path))

    if not model_files:
        raise FileNotFoundError(f"No matching v*_*evaluation*.jsonl files found in {chained_dir}")

    model_files.sort(key=lambda x: x[0])

    if sample_size is not None and sample_size < len(model_files):
        random.seed(42)
        model_files = sorted(random.sample(model_files, sample_size), key=lambda x: x[0])

    responses_by_version = {}
    for version_num, file_path in model_files:
        entries = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: failed to parse a line in {file_path}")
        responses_by_version[f"v{version_num}"] = entries

    return responses_by_version


def extract_subquestion_data_from_chained(responses_by_version, format_mapping):
    rows = []

    # Only score subquestions present in every version, to avoid spurious
    # N/A padding when a run silently dropped a question.
    questions_by_version = {
        version: {(r["overall_question_n"], r["question_n"]) for r in responses}
        for version, responses in responses_by_version.items()
    }
    common_questions = set.intersection(*questions_by_version.values()) if questions_by_version else set()
    print(f"Found {len(common_questions)} questions common to all {len(responses_by_version)} versions")

    # Serials come from expected_answer (ground truth), not from any one
    # model's response, so a badly-parsed answer can't hide/invent a serial.
    question_serials = {}
    for responses in responses_by_version.values():
        for response in responses:
            key = (response["overall_question_n"], response["question_n"])
            if key not in common_questions:
                continue
            question_serials.setdefault(key, set()).update(response.get("expected_answer", {}).keys())

    subquestions = sorted(
        (overall_q, question_n, serial)
        for (overall_q, question_n), serials in question_serials.items()
        for serial in serials
    )

    sorted_versions = sorted(responses_by_version.keys(), key=lambda v: int(v[1:]))

    for overall_question_n, question_n, serial in subquestions:
        row = {
            "questions": "",
            "overall_question_n": overall_question_n,
            "question_n": question_n,
            "serial": serial,
            "format": format_mapping.get(overall_question_n, "Unknown"),
            "correct_answer": [],
        }

        model_answers_list = []
        for version in sorted_versions:
            version_answer = "N/A"
            for response in responses_by_version[version]:
                if response["overall_question_n"] == overall_question_n and response["question_n"] == question_n:
                    if not row["questions"]:
                        context = response.get("question_details", {}).get("metadata", {}).get("context", "")
                        row["questions"] = context[:200] + "..." if context else ""
                    if not row["correct_answer"]:
                        expected = response.get("expected_answer", {})
                        if serial in expected:
                            row["correct_answer"] = parse_correct_answer(expected[serial])
                    model_parsed = response.get("model_parsed_response", {})
                    if serial in model_parsed:
                        version_answer = model_parsed[serial]
                    break

            row[f"model_answer_{version}"] = version_answer
            row[f"is_{version}_correct"] = check_answer_correctness(
                version_answer, row["correct_answer"], overall_question_n, question_n, serial
            )
            model_answers_list.append(version_answer)

        majority, tiebreaker, majority_size = calculate_majority_and_tiebreaker(model_answers_list)
        row["majority"] = majority
        row["tiebreaker"] = tiebreaker
        row["majority_size"] = majority_size
        row["unique_answers"] = unique_valid_answers(model_answers_list)
        correct = row["correct_answer"]
        row["is_majority_correct"] = check_answer_correctness(majority, correct, overall_question_n, question_n, serial)
        row["is_tiebreaker_correct"] = check_answer_correctness(tiebreaker, correct, overall_question_n, question_n, serial)
        row["is_any_correct"] = any(
            check_answer_correctness(a, correct, overall_question_n, question_n, serial) for a in model_answers_list
        )
        row["number_correct"] = sum(
            check_answer_correctness(a, correct, overall_question_n, question_n, serial) for a in model_answers_list
        )

        rows.append(row)

    return rows


def save_to_csv(rows, output_path):
    import csv

    if not rows:
        print("No data to save.")
        return

    columns = ["questions", "overall_question_n", "question_n", "serial", "format", "correct_answer"]
    versions = sorted({
        int(re.match(r"model_answer_v(\d+)", c).group(1))
        for c in rows[0] if re.match(r"model_answer_v(\d+)", c)
    })
    for v in versions:
        columns += [f"model_answer_v{v}", f"is_v{v}_correct"]
    columns += [
        "majority", "tiebreaker", "majority_size", "unique_answers",
        "is_majority_correct", "is_tiebreaker_correct", "is_any_correct", "number_correct",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


def main(model_name, sample_size, openrouter_runs_dir="openrouter_runs", output_dir="openrouter_analysis",
         format_mapping_csv=FORMAT_MAPPING_CSV):
    random.seed(42)  # one-time, for whole-run reproducibility -- see eval_common.calculate_majority_and_tiebreaker

    format_mapping = load_format_mapping(format_mapping_csv)

    model_name_lower = model_name.lower()
    all_rows = []
    for format_name in FORMATS:
        chained_dir = f"{openrouter_runs_dir}/{model_name}_chained_{format_name}"
        print(f"Loading chained prompt responses from {chained_dir}...")
        try:
            responses_by_version = load_chained_prompt_responses(chained_dir, sample_size)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        print(f"Loaded {len(responses_by_version)} versions for {format_name}: {sorted(responses_by_version, key=lambda v: int(v[1:]))}")
        all_rows.extend(extract_subquestion_data_from_chained(responses_by_version, format_mapping))

    if not all_rows:
        print("No data found across all formats.")
        return

    analysis_dir = Path(output_dir) / model_name_lower
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / f"chained_subquestion_eval_{model_name_lower}_all_formats_{sample_size}.csv"
    save_to_csv(all_rows, output_path)


if __name__ == "__main__":
    fire.Fire(main)
