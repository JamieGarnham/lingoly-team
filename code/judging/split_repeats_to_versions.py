#!/usr/bin/env python3
"""
Split a benchmark_model*.py --repeats N output file into the per-version
file layout (openrouter_runs/<subfolder>/vN_<name>.json) that
evaluate_subquestions.py expects.

In the original private research repo, each of the 32 repeated samples for
a model was saved to its own file by hand, one v1.json..v32.json per
repetition, under openrouter_runs/<model>/. benchmark_model*.py in this repo
instead writes all repetitions of a run to a single JSON file, with each
entry's 'repetition_idx' (0-indexed) recording which repetition it belongs
to. This script reverses that: it groups entries by repetition_idx and
writes one file per repetition, named to match the v(\\d+)_.*lingoly.*\\.json
pattern evaluate_subquestions.py's load_model_responses() globs for.

Usage:
    python split_repeats_to_versions.py ../../data/responses_obf/Sonnet_lingoly_rp32.json
    # -> writes openrouter_runs/Sonnet_lingoly_rp32/v1_Sonnet_lingoly_rp32.json ... v32_....json

    python evaluate_subquestions.py Sonnet_lingoly_rp32 32
    # (run from wherever your openrouter_runs/ directory lives -- see that
    # script's own docstring)
"""

import json
from collections import defaultdict
from pathlib import Path

import fire


def split_repeats(input_json, subfolder=None, openrouter_runs_dir="openrouter_runs"):
    input_path = Path(input_json)
    with open(input_path, encoding="utf-8") as f:
        entries = json.load(f)

    by_repetition = defaultdict(list)
    for entry in entries:
        by_repetition[entry.get("repetition_idx", 0)].append(entry)

    if len(by_repetition) <= 1:
        print(
            f"Warning: only found {len(by_repetition)} distinct repetition_idx value(s) in "
            f"{input_json} -- was this generated with --repeats > 1?"
        )

    if subfolder is None:
        subfolder = input_path.stem

    out_dir = Path(openrouter_runs_dir) / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem
    for repetition_idx in sorted(by_repetition):
        version_num = repetition_idx + 1  # v1, v2, ... (matches the original 1-indexed convention)
        out_path = out_dir / f"v{version_num}_{base_name}_lingoly.json"
        out_path.write_text(json.dumps(by_repetition[repetition_idx], indent=2, ensure_ascii=False))
        print(f"Wrote {len(by_repetition[repetition_idx])} questions to {out_path}")

    print(f"\nDone. {len(by_repetition)} version file(s) written to {out_dir}")
    print(f"Run: python evaluate_subquestions.py {subfolder} {len(by_repetition)}")


if __name__ == "__main__":
    fire.Fire(split_repeats)
