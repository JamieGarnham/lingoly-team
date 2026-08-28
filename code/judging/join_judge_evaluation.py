#!/usr/bin/env python3
"""
Join a subquestion_eval CSV (from evaluate_baseline.py / subsample_eval.py)
with the outputs of llm_judge_responses.py (reranking judge) and/or
llm_mc_responses.py (top-1 judge), adding judged_answer / is_judge_correct /
mc_judged_answer / is_mc_judge_correct columns.

This replaces join_judge_evaluation.py from the private research repo. That
version located its inputs by globbing a hardcoded absolute path
(/Users/jamiegarnham/lingoly2/...) that only existed on one machine and no
longer exists at all -- so instead of automatic discovery, this version
takes explicit paths for every input. Nothing is guessed or globbed; you
choose exactly which judge run's output to join against which subquestion
CSV.

Example:
    python join_judge_evaluation.py \\
        --subquestion_csv ../../data/judge_output/subquestion_eval_gemini_shuffle_4.csv \\
        --judge_jsonl ../../data/judge_output/Gemini_2.5_Flash_..._judge_evaluation_....jsonl \\
        --mc_judge_jsonl ../../data/mc_judge_output/Gemini_2.5_Flash_..._mc_judge_evaluation_....jsonl \\
        --output_csv ../../data/judge_output/subquestion_eval_gemini_shuffle_4_with_judge_evaluation.csv

Either judge_jsonl or mc_judge_jsonl (or both) may be omitted -- the
corresponding columns are then left blank rather than erroring out.
"""

import csv
import json
import sys
from pathlib import Path

import fire

from eval_common import check_answer_correctness, parse_correct_answer

csv.field_size_limit(sys.maxsize)


def classify_judge_validity(scores):
    """Classify a reranking judge's score set as 'reranker' (whole-number
    ranks with exactly one 1.0), 'scorer' (0-1 range, unique max), or
    'invalid'."""
    if not scores:
        return "invalid"
    try:
        numeric_scores = [float(s) for s in scores]
    except (TypeError, ValueError):
        return "invalid"

    if all(s == 0.0 for s in numeric_scores):
        return "invalid"

    if all(s > 0 and s == int(s) for s in numeric_scores):
        return "reranker" if numeric_scores.count(1.0) == 1 else "invalid"

    if all(0.0 <= s <= 1.0 for s in numeric_scores):
        max_score = max(numeric_scores)
        return "scorer" if numeric_scores.count(max_score) == 1 else "invalid"

    return "invalid"


def load_judge_evaluation(judge_jsonl_path):
    """key: (overall_question_n, question_n, serial) -> (judged_answer, judge_prompt, scores)"""
    lookup = {}
    with open(judge_jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["overall_question_n"]), row["question_n"].strip(), row["serial"].strip())

            judged_answer = ""
            scores = row.get("scores")
            if isinstance(scores, dict) and "unique_answers" in row:
                try:
                    score_values = list(scores.values())
                    validity = classify_judge_validity(score_values)
                    unique_answers = row["unique_answers"]
                    answer_index = None
                    if validity == "reranker":
                        for k, v in scores.items():
                            if v == 1.0 and k.startswith("answer_"):
                                answer_index = int(k.replace("answer_", "")) - 1
                                break
                    elif validity == "scorer":
                        max_score = max(scores.values())
                        for k, v in scores.items():
                            if v == max_score and k.startswith("answer_"):
                                answer_index = int(k.replace("answer_", "")) - 1
                                break
                    if answer_index is not None and 0 <= answer_index < len(unique_answers):
                        judged_answer = unique_answers[answer_index]
                except Exception:
                    judged_answer = ""

            lookup[key] = (judged_answer, row.get("judge_prompt", ""), scores)

    print(f"Loaded {len(lookup)} judge evaluations from {judge_jsonl_path}")
    return lookup


def load_mc_judge_evaluation(mc_judge_jsonl_path):
    """key: (overall_question_n, question_n, serial) -> (mc_judged_answer, mc_judge_prompt, mc_judge_validity)"""
    lookup = {}
    with open(mc_judge_jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["overall_question_n"]), row["question_n"].strip(), row["serial"].strip())

            mc_judged_answer, mc_judge_validity = "", False
            parse_result = row.get("parse_result", {})
            if parse_result.get("valid", False):
                mc_judged_answer = parse_result.get("selected_answer", "")
                mc_judge_validity = True

            lookup[key] = (mc_judged_answer, row.get("mc_judge_prompt", ""), mc_judge_validity)

    print(f"Loaded {len(lookup)} MC judge evaluations from {mc_judge_jsonl_path}")
    return lookup


def join(subquestion_csv, judge_jsonl=None, mc_judge_jsonl=None, output_csv=None):
    judge_lookup = load_judge_evaluation(judge_jsonl) if judge_jsonl else {}
    mc_judge_lookup = load_mc_judge_evaluation(mc_judge_jsonl) if mc_judge_jsonl else {}

    if not judge_jsonl and not mc_judge_jsonl:
        print("Warning: no judge_jsonl or mc_judge_jsonl given -- output will just copy the input CSV "
              "with empty judge columns.")

    rows_processed = judge_matches = mc_judge_matches = 0

    with open(subquestion_csv, encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + [
            "judged_answer", "judge_prompt", "is_judge_correct", "judge_validity", "is_judge_used",
            "mc_judged_answer", "mc_judge_prompt", "is_mc_judge_correct", "mc_judge_validity", "is_mc_judge_used",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                rows_processed += 1
                key = (int(row["overall_question_n"]), row["question_n"].strip(), row["serial"].strip())
                correct_answers_list = parse_correct_answer(row["correct_answer"])

                if key in judge_lookup:
                    judged_answer, judge_prompt, scores = judge_lookup[key]
                    row["judged_answer"] = judged_answer
                    row["judge_prompt"] = judge_prompt
                    row["is_judge_used"] = True
                    judge_matches += 1
                    row["judge_validity"] = classify_judge_validity(list(scores.values())) if isinstance(scores, dict) else "invalid"
                    row["is_judge_correct"] = (
                        check_answer_correctness(judged_answer, correct_answers_list, key[0], key[1], key[2])
                        if row["judge_validity"] != "invalid" and judged_answer else False
                    )
                else:
                    row["judged_answer"] = row["judge_prompt"] = row["judge_validity"] = ""
                    row["is_judge_correct"] = ""
                    row["is_judge_used"] = False

                if key in mc_judge_lookup:
                    mc_judged_answer, mc_judge_prompt, mc_judge_validity = mc_judge_lookup[key]
                    row["mc_judged_answer"] = mc_judged_answer
                    row["mc_judge_prompt"] = mc_judge_prompt
                    row["mc_judge_validity"] = mc_judge_validity
                    row["is_mc_judge_used"] = True
                    if mc_judge_validity and mc_judged_answer:
                        row["is_mc_judge_correct"] = check_answer_correctness(
                            mc_judged_answer, correct_answers_list, key[0], key[1], key[2]
                        )
                        mc_judge_matches += 1
                    else:
                        row["is_mc_judge_correct"] = False
                else:
                    row["mc_judged_answer"] = row["mc_judge_prompt"] = ""
                    row["mc_judge_validity"] = ""
                    row["is_mc_judge_correct"] = ""
                    row["is_mc_judge_used"] = False

                writer.writerow(row)

    print(f"Processed {rows_processed} rows: {judge_matches} judge matches, {mc_judge_matches} MC judge matches.")
    print(f"Output saved to: {output_csv}")


def main(subquestion_csv, judge_jsonl=None, mc_judge_jsonl=None, output_csv=None):
    if output_csv is None:
        stem = Path(subquestion_csv).stem
        output_csv = str(Path(subquestion_csv).parent / f"{stem}_with_judge_evaluation.csv")
    join(subquestion_csv, judge_jsonl, mc_judge_jsonl, output_csv)


if __name__ == "__main__":
    fire.Fire(main)
