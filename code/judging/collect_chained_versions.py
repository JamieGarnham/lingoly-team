#!/usr/bin/env python3
"""
Copy/rename a set of chained_prompting_{monolingual,pattern,rosetta}.py
output files into the openrouter_runs/<model>_chained_<format>/v{N}_....jsonl
layout evaluate_chained_subquestions.py expects.

chained_prompting_monolingual.py / _pattern.py / _rosetta.py each write one
timestamped file per invocation (no built-in repeat loop), so producing a
32-sample self-consistency dataset means running the script 32 times per
model per format and then collecting the resulting files -- this script is
that collection step, generalizing the private research repo's manual
renaming into something repeatable.

Handles chained_prompting_rosetta.py's inconsistent output filename (unlike
_pattern.py / _monolingual.py, it omits the format name --
'<model>_chained_evaluation_<timestamp>.jsonl' instead of
'<model>_chained_rosetta_evaluation_<timestamp>.jsonl') by matching either.

Usage:
    python collect_chained_versions.py --model_name Sonnet --format rosetta
    # -> copies every matching file in data/chained_responses/ into
    #    openrouter_runs/Sonnet_chained_rosetta/v1_..., v2_..., ... (sorted
    #    by the timestamp in the filename), one per file found.

    python collect_chained_versions.py --model_name Sonnet --format pattern --num_versions 32
    # -> errors out if fewer than 32 matching files are found, otherwise
    #    uses the first 32 by timestamp.
"""

import os
import re
import shutil
from pathlib import Path

import fire

FORMAT_TO_PATTERNS = {
    "monolingual": [r"^{model}_chained_monolingual_evaluation_.*\.jsonl$"],
    "pattern": [r"^{model}_chained_pattern_evaluation_.*\.jsonl$"],
    # rosetta's own script omits the format name from its filename, but a
    # manually-renamed file (as shipped in data/chained_responses.zip) may
    # include it -- match either.
    "rosetta": [
        r"^{model}_chained_evaluation_.*\.jsonl$",
        r"^{model}_chained_rosetta_evaluation_.*\.jsonl$",
    ],
}


def find_matching_files(input_dir, model_name, format_name):
    if format_name not in FORMAT_TO_PATTERNS:
        raise ValueError(f"format must be one of {list(FORMAT_TO_PATTERNS)}, got {format_name!r}")

    patterns = [re.compile(p.format(model=re.escape(model_name))) for p in FORMAT_TO_PATTERNS[format_name]]
    files = [
        f for f in os.listdir(input_dir)
        if any(p.match(f) for p in patterns)
    ]

    def extract_timestamp(filename):
        match = re.search(r"(\d{8}_\d{6})", filename)
        return match.group(1) if match else filename

    files.sort(key=extract_timestamp)
    return [os.path.join(input_dir, f) for f in files]


def main(model_name, format, input_dir="../../data/chained_responses", num_versions=None,
         output_dir=None):
    """
    model_name: the exact model_list.json key used when the chained-prompting
        script was run (this is what appears in its output filenames).
    format: one of 'monolingual', 'pattern', 'rosetta'.
    num_versions: how many files to use, earliest by timestamp first
        (default: use every matching file found).
    output_dir: defaults to 'openrouter_runs/<model_name>_chained_<format>'.
    """
    files = find_matching_files(input_dir, model_name, format)
    if not files:
        raise FileNotFoundError(f"No chained-prompting output files found for model={model_name!r} format={format!r} in {input_dir}")
    print(f"Found {len(files)} matching file(s)")

    if num_versions is None:
        num_versions = len(files)
    elif num_versions > len(files):
        raise ValueError(f"Requested {num_versions} versions but only {len(files)} files found")

    if output_dir is None:
        output_dir = f"openrouter_runs/{model_name}_chained_{format}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_versions):
        src = Path(files[i])
        dest = Path(output_dir) / f"v{i + 1}_{src.name}"
        shutil.copy2(src, dest)
        print(f"Copied {src.name} -> {dest.name}")

    print(f"\nDone. {num_versions} version file(s) written to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
