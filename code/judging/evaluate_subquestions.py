#!/usr/bin/env python3
"""
Evaluate model responses at the subquestion level -- faithful port of the
private research repo's evaluate_subquestions.py (the actual script used to
produce the paper's results tables), preserving its original file-layout
convention rather than benchmark_model.py's single --repeats N file.

Reads openrouter_runs/<subfolder>/vN_*lingoly*.json (one file per repeated
sample -- see split_repeats_to_versions.py to produce these from a
benchmark_model*.py --repeats N run; in the original private repo these
were saved individually by hand), and writes a CSV with one row per
subquestion: all N model answers, per-answer correctness, majority vote,
tiebreaker, and unique-answer set.

If you don't need byte-for-byte fidelity to the original file layout,
evaluate_baseline.py in this same folder does the equivalent scoring
directly from a single benchmark_model*.py output file -- both share the
same correctness logic in eval_common.py, so they agree on every row.

Usage:
    python evaluate_subquestions.py <subfolder> <sample_size>
    # reads openrouter_runs/<subfolder>/v*.json, randomly samples
    # <sample_size> of them (seeded, so reproducible), writes
    # openrouter_analysis/subquestion_eval_<subfolder>_<sample_size>.csv
"""

import csv
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


def load_format_mapping(path=FORMAT_MAPPING_CSV):
    """Format lookup keyed by overall_question_n.

    The original script read this from testing/data/past-exam-papers.csv
    (a much larger file with many unrelated columns). This repo ships the
    minimal overall_question_n -> format mapping extracted from that source
    instead (see code/dataset_creation/create_benchmark_same_obf.py), which
    covers the same 82 problems -- the values are identical, just read from
    a smaller file.
    """
    mapping = {}
    if not Path(path).exists():
        print(f"Warning: {path} not found. Using empty format mapping.")
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[int(row["overall_question_n"])] = row["format"]
    return mapping


def load_model_responses(subfolder, sample_size=None, openrouter_runs_dir="openrouter_runs"):
    openrouter_dir = Path(openrouter_runs_dir) / subfolder
    if not openrouter_dir.exists():
        raise FileNotFoundError(
            f"{openrouter_dir} does not exist. Run split_repeats_to_versions.py first "
            f"to produce it from a benchmark_model*.py --repeats N output file."
        )

    model_files = []
    for file_path in openrouter_dir.glob("v*_*lingoly*.json"):
        match = re.match(r"v(\d+)_.*lingoly.*\.json", file_path.name)
        if match:
            model_files.append((int(match.group(1)), file_path))

    if not model_files:
        raise FileNotFoundError(f"No matching v*_*lingoly*.json files found in {openrouter_dir}")

    model_files.sort(key=lambda x: x[0])

    if sample_size is not None and sample_size < len(model_files):
        random.seed(42)
        model_files = sorted(random.sample(model_files, sample_size), key=lambda x: x[0])

    responses_by_version = {}
    for version_num, file_path in model_files:
        with open(file_path, encoding="utf-8") as f:
            responses_by_version[f"v{version_num}"] = json.load(f)

    return responses_by_version


def extract_subquestion_data(responses_by_version, format_mapping):
    rows = []
    first_version = next(iter(responses_by_version))
    questions = responses_by_version[first_version]

    for question_data in questions:
        overall_question_n = question_data["overall_question_n"]
        question_n = question_data["question_n"]
        questions_text = question_data["questions"]
        question_format = format_mapping.get(overall_question_n, "Unknown")
        correct_answers = question_data["correct_answers"][0]

        for serial, correct_answer in correct_answers.items():
            correct_answers_list = parse_correct_answer(correct_answer)

            row = {
                "questions": questions_text,
                "overall_question_n": overall_question_n,
                "question_n": question_n,
                "serial": serial,
                "format": question_format,
                "correct_answer": correct_answer,
            }

            model_answers_list = []
            for version, version_responses in responses_by_version.items():
                version_question = next(
                    (q for q in version_responses
                     if q["overall_question_n"] == overall_question_n and q["question_n"] == question_n),
                    None,
                )
                model_answers = version_question.get("model_answers", {}) if version_question else {}
                answer = model_answers.get(serial, "N/A") if isinstance(model_answers, dict) else "N/A"

                row[f"model_answer_{version}"] = answer
                row[f"is_{version}_correct"] = check_answer_correctness(
                    answer, correct_answers_list, overall_question_n, question_n, serial
                )
                model_answers_list.append(answer)

            majority, tiebreaker, majority_size = calculate_majority_and_tiebreaker(model_answers_list)
            row["majority"] = majority
            row["tiebreaker"] = tiebreaker
            row["majority_size"] = majority_size
            row["unique_answers"] = unique_valid_answers(model_answers_list)
            row["is_majority_correct"] = check_answer_correctness(
                majority, correct_answers_list, overall_question_n, question_n, serial
            )
            row["is_tiebreaker_correct"] = check_answer_correctness(
                tiebreaker, correct_answers_list, overall_question_n, question_n, serial
            )
            row["is_any_correct"] = any(
                check_answer_correctness(a, correct_answers_list, overall_question_n, question_n, serial)
                for a in model_answers_list
            )
            row["number_correct"] = sum(
                check_answer_correctness(a, correct_answers_list, overall_question_n, question_n, serial)
                for a in model_answers_list
            )

            rows.append(row)

    return rows


def save_to_csv(rows, output_path):
    if not rows:
        print("No data to save.")
        return

    columns = ["questions", "overall_question_n", "question_n", "serial", "format", "correct_answer"]
    model_columns = sorted(
        (c for c in rows[0] if c.startswith("model_answer_v")),
        key=lambda c: int(c.split("model_answer_v")[1]),
    )
    for model_col in model_columns:
        version = model_col.replace("model_answer_", "")
        columns += [model_col, f"is_{version}_correct"]
    columns += [
        "majority", "tiebreaker", "majority_size", "unique_answers",
        "is_majority_correct", "is_tiebreaker_correct", "is_any_correct", "number_correct",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")


def main(subfolder, sample_size, openrouter_runs_dir="openrouter_runs", output_dir="openrouter_analysis",
         format_mapping_csv=FORMAT_MAPPING_CSV):
    random.seed(42)  # one-time, for whole-run reproducibility -- see eval_common.calculate_majority_and_tiebreaker

    format_mapping = load_format_mapping(format_mapping_csv)

    print(f"Loading model responses from {subfolder}...")
    responses_by_version = load_model_responses(subfolder, sample_size, openrouter_runs_dir)
    print(f"Loaded {len(responses_by_version)} versions: {list(responses_by_version.keys())}")

    rows = extract_subquestion_data(responses_by_version, format_mapping)

    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"subquestion_eval_{subfolder}_{sample_size}.csv"
    save_to_csv(rows, output_path)


if __name__ == "__main__":
    fire.Fire(main)
