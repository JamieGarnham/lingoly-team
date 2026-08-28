#!/usr/bin/env python3
"""
Recursively subsample the repetitions in an evaluate_baseline.py CSV and
recompute majority-vote / tiebreaker / correctness metrics at each size.

Ports the paper's inference-time budget simulation (§3.3): from an initial
sample of N responses per question (e.g. 32), take nested random subsamples
of N/2, N/4, ... down to 1, so that the subsample at each size is a subset of
the one above it (this makes the resulting accuracy-vs-budget curve
consistent rather than independently noisy at each size).
"""

import csv
import sys
from pathlib import Path

import fire

from eval_common import calculate_majority_and_tiebreaker, check_answer_correctness, parse_correct_answer, unique_valid_answers

import random


def detect_versions(row):
    versions = []
    i = 1
    while f"model_answer_v{i}" in row:
        versions.append(f"v{i}")
        i += 1
    return versions


def default_subsample_sizes(n_versions):
    sizes = []
    size = n_versions // 2
    while size >= 1:
        sizes.append(size)
        size //= 2
    return sizes


def nested_subsample_chain(all_versions, sizes, seed=42):
    """Each size's version set is a random subset of the previous (larger)
    size's set, so smaller budgets are nested within larger ones."""
    random.seed(seed)
    chain = {}
    current = list(all_versions)
    for size in sizes:
        if size < len(current):
            current = sorted(random.sample(current, size), key=lambda v: int(v[1:]))
        chain[size] = current
    return chain


def recompute_row(row, versions, correct_answers_list, overall_question_n, question_n, serial):
    model_answers = [row.get(f"model_answer_{v}", "N/A") for v in versions]

    out = {}
    for v, answer in zip(versions, model_answers):
        out[f"model_answer_{v}"] = answer
        out[f"is_{v}_correct"] = check_answer_correctness(
            answer, correct_answers_list, overall_question_n, question_n, serial
        )

    majority, tiebreaker, majority_size = calculate_majority_and_tiebreaker(model_answers)
    out["majority"] = majority
    out["tiebreaker"] = tiebreaker
    out["majority_size"] = majority_size
    out["unique_answers"] = unique_valid_answers(model_answers)
    out["is_majority_correct"] = check_answer_correctness(
        majority, correct_answers_list, overall_question_n, question_n, serial
    )
    out["is_tiebreaker_correct"] = check_answer_correctness(
        tiebreaker, correct_answers_list, overall_question_n, question_n, serial
    )
    out["is_any_correct"] = any(
        check_answer_correctness(a, correct_answers_list, overall_question_n, question_n, serial)
        for a in model_answers
    )
    out["number_correct"] = sum(
        check_answer_correctness(a, correct_answers_list, overall_question_n, question_n, serial)
        for a in model_answers
    )
    return out


def process_subsample(input_file, output_prefix, subsample_sizes=None):
    csv.field_size_limit(sys.maxsize)

    print(f"Loading {input_file}...")
    with open(input_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows")

    if not rows:
        print("No rows to process.")
        return

    all_versions = detect_versions(rows[0])
    print(f"Found {len(all_versions)} model versions: {all_versions}")

    if subsample_sizes is None:
        subsample_sizes = default_subsample_sizes(len(all_versions))
    print(f"Subsample sizes: {subsample_sizes}")

    version_chain = nested_subsample_chain(all_versions, subsample_sizes)

    for size in subsample_sizes:
        selected_versions = version_chain[size]
        print(f"Processing subsample size {size} (versions: {selected_versions})...")
        output_rows = []

        for row in rows:
            overall_question_n = int(row["overall_question_n"])
            question_n = row["question_n"]
            serial = row["serial"]
            correct_answers_list = parse_correct_answer(row["correct_answer"])

            output_row = {
                "questions": row["questions"],
                "overall_question_n": overall_question_n,
                "question_n": question_n,
                "serial": serial,
                "format": row["format"],
                "correct_answer": row["correct_answer"],
            }
            output_row.update(
                recompute_row(row, selected_versions, correct_answers_list, overall_question_n, question_n, serial)
            )
            output_rows.append(output_row)

        output_file = f"{output_prefix}_{size}.csv"
        columns = ["questions", "overall_question_n", "question_n", "serial", "format", "correct_answer"]
        for v in selected_versions:
            columns += [f"model_answer_{v}", f"is_{v}_correct"]
        columns += [
            "majority", "tiebreaker", "majority_size", "unique_answers",
            "is_majority_correct", "is_tiebreaker_correct", "is_any_correct", "number_correct",
        ]

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"  Wrote {len(output_rows)} rows to {output_file}")

    print("Completed all subsamples!")


def main(input_file, output_prefix=None, subsample_sizes=None):
    """
    input_file: CSV produced by evaluate_baseline.py.
    output_prefix: defaults to '<input stem without trailing _N>' next to the
        input file; writes '<output_prefix>_<size>.csv' for each size.
    subsample_sizes: comma-separated sizes (e.g. "16,8,4,2,1"); defaults to
        halving the detected number of repetitions down to 1.
    """
    if isinstance(subsample_sizes, str):
        subsample_sizes = [int(s) for s in subsample_sizes.split(",")]

    if output_prefix is None:
        input_path = Path(input_file)
        stem = input_path.stem
        output_prefix = str(input_path.parent / stem)

    process_subsample(input_file, output_prefix, subsample_sizes)


if __name__ == "__main__":
    fire.Fire(main)
