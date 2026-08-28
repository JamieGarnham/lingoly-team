"""Shared answer-normalization and correctness-checking logic for the
judging pipeline (evaluate_baseline.py, subsample_eval.py,
join_judge_evaluation.py). Kept in one place so the three manual
corrections below (see Table 1 of the paper) aren't duplicated three times.
"""

import ast
import json
import random
from collections import Counter


def normalize_answer(answer):
    if not isinstance(answer, str):
        answer = str(answer)
    return answer.strip().rstrip(".").lower()


def is_valid_model_answer(answer):
    if not answer or answer == "N/A":
        return False
    if not isinstance(answer, str):
        answer = str(answer)
    answer_lower = answer.lower().strip()
    if "invalid json" in answer_lower or answer_lower == "":
        return False
    return True


def parse_correct_answer(answer_value):
    """Correct answers may be a plain string, a list, or a JSON/Python-literal
    string encoding a list (e.g. '["a", "b"]')."""
    if isinstance(answer_value, list):
        return answer_value
    if not isinstance(answer_value, str):
        return [str(answer_value)]
    if answer_value.startswith("[") and answer_value.endswith("]"):
        try:
            return json.loads(answer_value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(answer_value)
            except Exception:
                return [answer_value]
    return [answer_value]


def check_answer_correctness(answer, correct_answers, overall_question_n=None, question_n=None, serial=None):
    """Exact-match correctness check, with manual corrections to the dataset
    solutions (see paper Table 1 / README code/dataset_creation notes)."""
    if not is_valid_model_answer(answer):
        return False

    normalized_answer = normalize_answer(answer)

    # Correction 1: Q5.1(a) accepts any string containing both loanwords.
    if overall_question_n == 5 and question_n == "Q 5.1" and serial == "a":
        if "üpgontüd" in normalized_answer and "sopostüd" in normalized_answer:
            return True

    # Correction 2: Q170 Q5.(k) accepts either apostrophe glyph.
    if overall_question_n == 170 and question_n == "Q 5." and serial == "k":
        straight_apos, curly_apos = "'", chr(8217)
        if normalized_answer in (
            f"langgbu{straight_apos}", f"langgbu{curly_apos}",
            f"maysu{straight_apos}", f"maysu{curly_apos}",
        ):
            return True

    # Correction 3: Q75 Q7.(3) also accepts the spelled-out equivalent of "(2n)".
    if overall_question_n == 75 and question_n == "Q 7." and serial == "3":
        if "two people who are not siblings" in normalized_answer:
            return True

    normalized_correct = [normalize_answer(ca) for ca in correct_answers]
    return normalized_answer in normalized_correct


def calculate_majority_and_tiebreaker(model_answers):
    """Returns (majority, tiebreaker, majority_size). Ties are broken via
    random.choice(), with majority left as 'N/A' to record the tie.

    Deliberately does NOT seed the RNG here: this is called once per
    subquestion (potentially thousands of times per script run), and
    reseeding to the same fixed value before every call would make
    random.choice() deterministic across calls -- for any 2-way tie it
    would pick the same list position (e.g. always the first answer)
    every single time, rather than actually varying by row. Seed once,
    at the top of the calling script's entry point, for whole-run
    reproducibility instead."""
    valid_answers = []
    for ans in model_answers:
        if is_valid_model_answer(ans):
            valid_answers.append(str(ans) if isinstance(ans, (list, dict)) else ans)

    if not valid_answers:
        return "N/A", "N/A", 0

    answer_counts = Counter(valid_answers)
    max_count = max(answer_counts.values())
    majority_answers = [a for a, c in answer_counts.items() if c == max_count]

    if len(majority_answers) == 1:
        majority = tiebreaker = majority_answers[0]
    else:
        majority = "N/A"
        tiebreaker = random.choice(majority_answers)

    return majority, tiebreaker, max_count


def unique_valid_answers(model_answers):
    seen, unique = set(), []
    for ans in model_answers:
        if is_valid_model_answer(ans):
            ans_str = ans if isinstance(ans, str) else str(ans)
            if ans_str not in seen:
                seen.add(ans_str)
                unique.append(ans_str)
    return unique
