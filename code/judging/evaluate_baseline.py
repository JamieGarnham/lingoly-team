#!/usr/bin/env python3
"""
Evaluate a benchmark_model*.py response file at the subquestion level.

Ports the paper's baseline-scoring step (§3.2 exact match, §4.2.1 majority
vote / self-consistency) from the private research repo's
evaluate_subquestions.py. That version expected each of the 32 repeated
samples to live in its own file under openrouter_runs/<model>/vN.json; this
version instead reads the single JSON file that benchmark_model*.py actually
produces when run with --repeats 32, where all 32 repetitions of a question
appear as separate list entries sharing the same
(overall_question_n, question_n) and distinguished by 'repetition_idx'.

Output is a CSV with one row per subquestion, with model_answer_v1..vN /
is_vN_correct columns (N = number of repetitions found) plus majority-vote /
tiebreaker / unique-answer columns -- feed this into subsample_eval.py to
simulate smaller inference-time budgets, and llm_judge_responses.py /
llm_mc_responses.py to run LLM-as-a-judge on the same data.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import fire

from eval_common import (
    calculate_majority_and_tiebreaker,
    check_answer_correctness,
    parse_correct_answer,
    unique_valid_answers,
)

FORMAT_MAPPING_CSV = "../../data/overall_question_format_mapping.csv"


def load_format_mapping(path):
    mapping = {}
    if not Path(path).exists():
        print(f"Warning: {path} not found. 'format' column will be 'Unknown'.")
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[int(row["overall_question_n"])] = row["format"]
    return mapping


def load_repeated_responses(input_json):
    """Group a benchmark_model*.py output file by (overall_question_n,
    question_n), returning {key: [entries sorted by repetition_idx]}."""
    with open(input_json, encoding="utf-8") as f:
        entries = json.load(f)

    grouped = defaultdict(list)
    for entry in entries:
        key = (entry["overall_question_n"], entry["question_n"])
        grouped[key].append(entry)

    for key in grouped:
        grouped[key].sort(key=lambda e: e.get("repetition_idx", 0))

    return grouped


def extract_subquestion_rows(grouped, format_mapping):
    rows = []
    for (overall_question_n, question_n), entries in grouped.items():
        questions_text = entries[0]["questions"]
        question_format = format_mapping.get(overall_question_n, "Unknown")
        # correct_answers is a list containing one {serial: answer} dict
        correct_answers_by_serial = entries[0]["correct_answers"][0]

        n_repeats = len(entries)

        for serial, correct_answer in correct_answers_by_serial.items():
            correct_answers_list = parse_correct_answer(correct_answer)

            model_answers = []
            for entry in entries:
                answer = entry.get("model_answers", {}).get(serial, "N/A")
                model_answers.append(answer)

            row = {
                "questions": questions_text,
                "overall_question_n": overall_question_n,
                "question_n": question_n,
                "serial": serial,
                "format": question_format,
                "correct_answer": correct_answer,
            }

            for i, answer in enumerate(model_answers, start=1):
                row[f"model_answer_v{i}"] = answer
                row[f"is_v{i}_correct"] = check_answer_correctness(
                    answer, correct_answers_list, overall_question_n, question_n, serial
                )

            majority, tiebreaker, majority_size = calculate_majority_and_tiebreaker(model_answers)
            row["majority"] = majority
            row["tiebreaker"] = tiebreaker
            row["majority_size"] = majority_size
            row["unique_answers"] = unique_valid_answers(model_answers)
            row["is_majority_correct"] = check_answer_correctness(
                majority, correct_answers_list, overall_question_n, question_n, serial
            )
            row["is_tiebreaker_correct"] = check_answer_correctness(
                tiebreaker, correct_answers_list, overall_question_n, question_n, serial
            )
            row["is_any_correct"] = any(
                check_answer_correctness(a, correct_answers_list, overall_question_n, question_n, serial)
                for a in model_answers
            )
            row["number_correct"] = sum(
                check_answer_correctness(a, correct_answers_list, overall_question_n, question_n, serial)
                for a in model_answers
            )
            row["n_repeats"] = n_repeats

            rows.append(row)

    return rows


def save_to_csv(rows, output_path):
    if not rows:
        print("No data to save.")
        return

    n_repeats = rows[0]["n_repeats"]
    columns = ["questions", "overall_question_n", "question_n", "serial", "format", "correct_answer"]
    for i in range(1, n_repeats + 1):
        columns += [f"model_answer_v{i}", f"is_v{i}_correct"]
    columns += [
        "majority", "tiebreaker", "majority_size", "unique_answers",
        "is_majority_correct", "is_tiebreaker_correct", "is_any_correct", "number_correct",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows ({n_repeats} repetition(s) per subquestion) to {output_path}")


def main(input_json, output_csv=None, format_mapping_csv=FORMAT_MAPPING_CSV):
    """
    input_json: a benchmark_model.py / _probabilities.py / _shuffle.py output
        file (run with --repeats N for a meaningful self-consistency sample).
    output_csv: defaults to '<input_json stem>_subquestion_eval.csv' next to
        the input file.
    """
    random.seed(42)  # one-time, for whole-run reproducibility -- see eval_common.calculate_majority_and_tiebreaker

    input_path = Path(input_json)
    if output_csv is None:
        output_csv = input_path.with_name(f"{input_path.stem}_subquestion_eval.csv")

    format_mapping = load_format_mapping(format_mapping_csv)

    print(f"Loading {input_json}...")
    grouped = load_repeated_responses(input_json)
    print(f"Loaded {len(grouped)} questions")

    rows = extract_subquestion_rows(grouped, format_mapping)
    save_to_csv(rows, output_csv)


if __name__ == "__main__":
    fire.Fire(main)
