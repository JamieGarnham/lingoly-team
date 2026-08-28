# Could language models win the Linguistics Olympiad?

https://aclanthology.org/2026.conll-main.28/ presented at CoNLL 2026

## Revisions to paper

26/08/2026: updated 'Majority vote' and 'Upper bound' values for Gemini 2.5 Flash (K=2 and K=4) in final table of results.

## Revisions to repository

28/08/2026: Fixed three issues found while dry-running the reproduction steps below with synthetic model responses (no API calls):
- `code/judging/eval_common.py`'s Table 1 "Correction 1" (Q5.1(a) accepting
  any string containing both loanwords) silently never matched: the dataset
  stores accented characters in NFD (decomposed) form, while the hardcoded
  literals in that file are NFC (composed), so the `in` substring checks
  always failed. `normalize_answer()` now NFC-normalizes before comparing.
  This only affects the correctness of that one manually-corrected
  subquestion; it does not change any of the headline results in the paper,
  which were scored before this fix existed.
- `data/splits/benchmark_same_obf.jsonl.zip` and
  `benchmark_same_obf_rosetta.jsonl.zip` shipped with every entry's `format`
  field set to `null`, rather than the Rosetta/Pattern/Monolingual/Match-up
  value `create_benchmark_same_obf.py` actually looks up and writes today.
  Both files have been regenerated with the current scripts; verified
  byte-for-byte identical to the previous shipped files in every field
  *except* `format` (same 173/107 obfuscation selections, same seed). This
  field is otherwise unused downstream — `evaluate_baseline.py` and friends
  look up format independently from `overall_question_format_mapping.csv` —
  so this had no effect on any reported result, only on anyone reading the
  split file's own `format` field directly.
- `requirements.txt` was unpinned and, in practice, did not resolve to a
  working environment via `pip install -r requirements.txt` — `guidance`
  pulls in an exact `llguidance` version pip won't always satisfy on its
  own, alongside several other transitively-missing packages (see the
  comment in `requirements.txt`). All versions are now pinned to a
  combination verified to install and run cleanly, including the
  API-key-free `--generate_only` sanity check.

## Repository

This repository contains the benchmark data and code needed to reproduce the
results in the paper. It covers three stages: (1) building the benchmark
splits from the raw obfuscated question pool, (2) running language models
against those splits, and (3) evaluating the responses.

## Contamination protection

All benchmark data in `data/splits/` is shipped as password-protected zip
archives (`.jsonl.zip`), rather than plain text, to make the questions harder
to scrape and index verbatim by search engines / crawlers that end up as LLM
training data.

**Password for every archive in this repository: `olympiad`**

This is a single shared password for all `.zip` files under `data/`. It is
not a secret — it's published here deliberately — it just adds a barrier
against automated scraping. The code decrypts these archives automatically
(see below); you only need the password if you want to inspect a file
manually:

```bash
unzip -P olympiad data/splits/benchmark_same_obf.jsonl.zip -d /tmp/
```

or in Python:

```python
import pyzipper
with pyzipper.AESZipFile("data/splits/benchmark_same_obf.jsonl.zip") as zf:
    data = zf.read("benchmark_same_obf.jsonl", pwd=b"olympiad")
```

`data/responses_obf.zip` and `data/chained_responses.zip` (pre-generated
model outputs from the paper) are plain, unprotected zips — just compressed
for size, since they're model outputs rather than benchmark questions.

`data/results/*.zip`, however, **are** password-protected with the same
`olympiad` password: unlike the two files above, these contain full
scored results tables with a `correct_answer` column sitting right next to
every model's answers, which would defeat the point of the contamination
protection above if left in plain text.

## Repository layout

```
team_public/
├── requirements.txt
├── code/
│   ├── dataset_creation/     # build the benchmark splits in data/splits/
│   │   ├── create_benchmark_same_obf.py
│   │   ├── create_benchmark_single_obf_dev.py
│   │   ├── create_benchmark_single_obf_train.py
│   │   ├── filter_rosetta_problems.py
│   │   └── zip_io.py         # shared password-zip read/write helper
│   ├── benchmarking/         # run a model against a benchmark split
│   │   ├── benchmark_model.py
│   │   ├── benchmark_model_probabilities.py
│   │   ├── benchmark_model_shuffle.py
│   │   ├── load_questions*.py
│   │   ├── prompt_models*.py
│   │   └── shuffle_utils.py
│   ├── chained_prompting/    # multi-turn prompting + evaluation
│   │   ├── chained_prompting_{matchup,monolingual,pattern,rosetta}.py
│   │   ├── demo_chained_prompts.py
│   │   └── evaluate_chained_responses.py
│   └── judging/              # baseline scoring, self-consistency, LLM-as-a-judge
│       ├── evaluate_baseline.py             # scores a single --repeats N file directly
│       ├── evaluate_subquestions.py         # faithful port; needs openrouter_runs/ layout
│       ├── split_repeats_to_versions.py     # bridges the two: builds that layout
│       ├── subsample_eval.py                # inference-time budget simulation (32->1)
│       ├── evaluate_chained_subquestions.py # same, but for chained-prompting self-consistency
│       ├── collect_chained_versions.py      # builds the openrouter_runs/ layout for chained runs
│       ├── evaluate_matchup_multi_run.py    # Tree-of-Matches self-consistency (self-contained)
│       ├── llm_judge_responses.py
│       ├── llm_mc_responses.py
│       ├── join_judge_evaluation.py
│       └── eval_common.py    # shared answer-normalization / correctness logic
└── data/
    ├── model_list.json               # model registry (see below)
    ├── overall_question_format_mapping.csv
    ├── past-exam-papers.csv
    ├── splits/                       # password-protected benchmark data
    ├── responses_obf.zip             # pre-generated single-shot results (raw model output, no scoring)
    ├── chained_responses.zip         # pre-generated chained-prompting results (raw model output, no scoring)
    └── results/                      # scored results tables (password-protected, see below)
        ├── baseline_shuffle.zip
        ├── chained_prompting.zip
        ├── matchup.zip
        └── ablation_minimal_prompt.zip
```

All scripts use paths relative to their own directory (e.g. `../../data/...`),
so **run each script from inside the folder it lives in**:

```bash
cd code/benchmarking
python benchmark_model.py --model Sonnet --test_data_zip ../../data/splits/benchmark_same_obf.jsonl.zip
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes `torch`/`transformers`/`bitsandbytes`/`guidance`
for running local, open-weight models, plus their own transitive
dependencies (`llguidance`, `guidance-stitch`, `psutil`, `jinja2`,
`networkx`, `sympy`, `safetensors`, `typer`) pinned explicitly — a bare
`pip install guidance`/`transformers` does not reliably resolve these on
its own. If you only need closed APIs (OpenAI, Anthropic, Google, Cohere,
OpenRouter), you can comment all ten of those out — they're the heaviest
dependencies, and every version above them has been verified to install
cleanly together in a fresh virtualenv.

Note that `benchmark_model.py` / `benchmark_model_probabilities.py` /
`benchmark_model_shuffle.py` import `torch`/`transformers`/`guidance` at
the top of the file unconditionally, so even the API-key-free
`--generate_only` sanity check below needs the local-inference packages
installed too, regardless of which model you actually run.

### API keys

Set whichever of these your `model_list.json` entries need, as environment
variables:

| Env var              | Used for                                   |
|-----------------------|---------------------------------------------|
| `OPENAI_API_KEY`      | OpenAI models (read implicitly by the SDK)  |
| `ANTHROPIC_API_KEY`   | Claude models (read implicitly by the SDK)  |
| `COHERE_API_KEY`      | Cohere models (read implicitly by the SDK)  |
| `GOOGLE_API_KEY`      | Gemini models                               |
| `DEEPSEEK_API_KEY`    | DeepSeek models called directly             |
| `OPEN_ROUTER_API`     | Any model routed through OpenRouter         |
| `HF_TOKEN`            | Gated Hugging Face models (local inference) |

## Reproducing results

### 1. Dataset creation (optional)

The splits in `data/splits/` are already built — you don't need to run this
stage to use the benchmark. Run it only if you want to regenerate the splits
from the raw obfuscation pool (`data/splits/benchmark.jsonl.zip`, ~1,200
obfuscated question variants):

```bash
cd code/dataset_creation
python create_benchmark_same_obf.py      # -> data/splits/benchmark_same_obf.jsonl.zip
python create_benchmark_single_obf_dev.py    # -> benchmark_single_obf_dev.jsonl.zip
python create_benchmark_single_obf_train.py  # -> benchmark_single_obf_train.jsonl.zip
python filter_rosetta_problems.py            # -> benchmark_same_obf_rosetta.jsonl.zip
```

`create_benchmark_same_obf.py` randomly selects one obfuscation per problem
(seeded, `random.seed(42)`, so it's reproducible) and tags each entry with a
`format` field (Rosetta / Pattern / Monolingual / Match-up) looked up from
`data/overall_question_format_mapping.csv`. Note: only `benchmark_same_obf`,
`benchmark_single_obf_{dev,train}` and `benchmark_same_obf_rosetta` can be
regenerated from data included in this repo — the Pattern / Monolingual /
Match-up splits and the `benchmark_no_matchup.jsonl.zip` / `_rosetta.jsonl.zip`
files ship as-is with no generator script (see **Known limitations** below).

**Shuffle split (`benchmark_same_obf_shuffle.jsonl.zip`)**: this is not
produced by a script. Each problem sheet was manually inspected to determine
which sections of its context (if any) are order-insensitive (e.g. an
unordered vocabulary list, safe to shuffle) versus order-sensitive (e.g. a
numbered example sequence, must stay fixed), and `<SHUFFLE_START>...
<SHUFFLE_END>` / `<REORDER_START>...<REORDER_END>` markers were inserted by
hand around the shufflable sections. `shuffle_utils.py` /
`load_questions_shuffle.py` / `benchmark_model_shuffle.py` consume those
markers to reorder the enclosed content at inference time (see step 2) — but
recreating the annotated file itself from a clean `benchmark_same_obf.jsonl`
requires redoing that manual review, not running a script.

### 2. Running the benchmark

```bash
cd code/benchmarking
python benchmark_model.py \
    --model Sonnet \
    --test_data_zip ../../data/splits/benchmark_same_obf.jsonl.zip
```

- `benchmark_model.py` — single-shot benchmarking (supports `--cot`,
  `--no_context`, `--adversarial`, `--minimal_prompt`, `--repeats`, etc.)
- `benchmark_model_probabilities.py` — same, plus extracts per-answer
  confidence from token logprobs (`--calibrate_probs` for sigmoid
  calibration)
- `benchmark_model_shuffle.py` — targets the context-shuffled split
  (`benchmark_same_obf_shuffle.jsonl.zip` by default)

`--model` must match a key in `data/model_list.json`. Results are written to
`data/responses_obf/` (created automatically), named after that key's
`name` field (or `chkpoint` if set) — **not** the `--model` key itself.
For example, `--model Sonnet` (whose `model_list.json` entry has
`"name": "claude-3-5-sonnet-20241022"`) writes
`claude-3-5-sonnet-20241022_lingoly.json`, not `Sonnet_lingoly.json`.
The exact filename is also printed to stdout while the script runs
(`Looking for cache for <name>_lingoly...`); check that if unsure before
running step 5's commands, since they need the real filename to find the
file.

Pass `--generate_only True` to build and cache prompts without calling any
model — useful for a quick, API-key-free sanity check that the data decrypts
and loads correctly.

For a meaningful self-consistency / LLM-as-a-judge sample (§3.3 of the
paper), run with `--repeats 32` (or another power of two) — see step 5 for
scoring this output.

### 3. Chained (multi-turn) prompting

One script per problem format:

```bash
cd code/chained_prompting
python chained_prompting_rosetta.py --model_names Sonnet,GPT_4o --max_problems 5
python chained_prompting_pattern.py --model_names Sonnet
python chained_prompting_monolingual.py --model_names Sonnet
python chained_prompting_matchup.py --model_names Sonnet
```

Each reads the corresponding `data/splits/benchmark_same_obf_<format>.jsonl.zip`,
asks a first "reasoning" prompt per problem sheet, then asks each
subquestion in the same conversation. Results are written to
`data/chained_responses/` as timestamped JSONL files.

`demo_chained_prompts.py` prints the prompts for the first problem without
calling any model — useful for sanity-checking prompt formatting.

### 4. Evaluation

```bash
cd code/chained_prompting
python evaluate_chained_responses.py <path-to-chained-response.jsonl> \
    --output_csv results.csv --summary_json summary.json
```

Scores `model_parsed_response` against `expected_answer` with
Unicode-normalized matching.

### 5. Baseline scoring, self-consistency, and LLM-as-a-judge

`code/judging/` covers the rest of the paper's evaluation methodology (§3.2
exact match, §3.3 inference-time budget simulation, §4.2 aggregation
methods) for `benchmark_model*.py` output. This is a from-scratch port of
scripts that only ever existed in the private research repo, adapted to
work with this repo's actual file layout and conventions rather than that
repo's machine-specific paths — see the docstring in each file for exactly
what changed.

```bash
# 1. Generate a self-consistency sample (see step 2), e.g.:
cd code/benchmarking
python benchmark_model.py --model Sonnet --repeats 32 \
    --test_data_zip ../../data/splits/benchmark_same_obf.jsonl.zip

# The output filename is derived from model_list.json's "name" (or
# "chkpoint") field for --model, NOT the --model key itself -- for
# "Sonnet" that's "claude-3-5-sonnet-20241022", not "Sonnet" (see step 2
# above). Capture it once so the rest of this block is copy-pasteable:
OUT_NAME=claude-3-5-sonnet-20241022_lingoly_rp32

# 2. Score it at the subquestion level (majority vote, tiebreaker, exact match)
cd ../judging
python evaluate_baseline.py ../../data/responses_obf/$OUT_NAME.json

# 2 (alternative). To instead run the *original* evaluate_subquestions.py
# script byte-for-byte as it ran in the private repo -- e.g. if you need to
# cross-check against how the paper's numbers were actually produced --
# first split the single --repeats output into the per-version file layout
# it expects:
python split_repeats_to_versions.py ../../data/responses_obf/$OUT_NAME.json
python evaluate_subquestions.py $OUT_NAME 32
# Both paths share the same scoring logic (eval_common.py) and produce
# identical majority/tiebreaker/correctness columns -- verified against a
# synthetic multi-repetition sample. Pick evaluate_baseline.py for less
# setup, or the split + evaluate_subquestions.py pair for maximum fidelity
# to the original file layout.

# 3. Simulate smaller inference-time budgets (32 -> 16 -> 8 -> 4 -> 2 -> 1)
python subsample_eval.py ../../data/responses_obf/${OUT_NAME}_subquestion_eval.csv

# 4. LLM-as-a-judge: reranking judge (Figure 12) and/or top-1/MC judge (Figure 13)
python llm_judge_responses.py --model GPT_4o \
    --response_data_csv ../../data/responses_obf/${OUT_NAME}_subquestion_eval_8.csv
python llm_mc_responses.py --model GPT_4o \
    --response_data_csv ../../data/responses_obf/${OUT_NAME}_subquestion_eval_8.csv

# 5. Join the judge output back onto the subquestion CSV
python join_judge_evaluation.py \
    --subquestion_csv ../../data/responses_obf/${OUT_NAME}_subquestion_eval_8.csv \
    --judge_jsonl ../../data/judge_output/<judge output>.jsonl \
    --mc_judge_jsonl ../../data/mc_judge_output/<mc judge output>.jsonl
```

Notes:
- `evaluate_baseline.py` reads whatever `benchmark_model*.py` actually
  produces with `--repeats N` (all N repetitions of a question in one file,
  distinguished by `repetition_idx`) rather than the private repo's
  convention of N separate files under `openrouter_runs/<model>/vN.json` —
  so there's no `openrouter_runs/` folder to set up.
- It looks up each question's format from `data/overall_question_format_mapping.csv`
  (the same file `create_benchmark_same_obf.py` uses), rather than the
  private repo's `question_formats_fixed.csv` correction file, which doesn't
  exist here and was folded into that mapping already being correct.
- The judge scripts only prompt for subquestions with 2+ unique answers by
  default (`--min_unique_answers`) — a judge has nothing to rank/select
  between when every sample agreed.
- **Fixing the hardcoded path**: the private repo's `join_judge_evaluation.py`
  located its inputs by globbing a hardcoded absolute path
  (`/Users/jamiegarnham/lingoly2/...`) that only ever existed on one machine.
  This version takes `--subquestion_csv`, `--judge_jsonl`, and
  `--mc_judge_jsonl` as explicit arguments instead — nothing is auto-discovered,
  so there's no path to break when this repo is cloned elsewhere. Similarly,
  the private repo's `combine_judge_files.py` (which merged per-format judge
  runs back together) is no longer needed, since `llm_judge_responses.py` /
  `llm_mc_responses.py` process a whole CSV in one pass rather than being run
  once per format.

### 6. Self-consistency for chained prompting and Tree-of-Matches

The `code/judging/` self-consistency machinery above only covers
`benchmark_model*.py`'s single-shot output. Chained prompting (step 3) has
its own equivalent, since its answers live in a different JSONL shape
(`model_parsed_response` / `expected_answer` rather than `model_answers` /
`correct_answers`), and Match-up needs its own scorer entirely (its answers
are pairings, not simple strings).

Both require **actually running the relevant chained-prompting script
multiple times** (e.g. 32x) per model — there's no `--repeats` flag for
these like there is for `benchmark_model.py`, so this is real API cost, not
just a script you can run once.

```bash
cd code/chained_prompting
# Run 32 times per model per format (only rosetta/pattern/monolingual use
# eval_common.py's scoring; matchup is handled separately below):
for i in $(seq 1 32); do python chained_prompting_rosetta.py --model_names Sonnet; done

cd ../judging
# Collect those 32 raw output files into the layout evaluate_chained_subquestions.py expects
python collect_chained_versions.py --model_name Sonnet --format rosetta
# ... repeat collect_chained_versions.py for --format pattern and --format monolingual

# Score at the subquestion level, majority vote, combine all 3 formats into one CSV
python evaluate_chained_subquestions.py Sonnet 32
# Feed the resulting CSV into subsample_eval.py / llm_judge_responses.py /
# llm_mc_responses.py exactly as in step 5 above.

# Match-up (Tree-of-Matches) is separate and self-contained -- it
# auto-discovers and auto-versions chained_prompting_matchup.py's raw output
# itself, no collect_chained_versions.py step needed:
cd ../chained_prompting
for i in $(seq 1 32); do python chained_prompting_matchup.py --model_names Sonnet; done
cd ../judging
python evaluate_matchup_multi_run.py --model_name Sonnet --num_versions 32
```

Notes:
- `evaluate_chained_subquestions.py` reuses `eval_common.py`, so it agrees
  with `evaluate_baseline.py` / `evaluate_subquestions.py` on every
  correctness rule (including the three manual dataset corrections).
  `evaluate_matchup_multi_run.py` does not -- Match-up answers are pairings
  extracted from `confirmed_matches`, not free-text strings, so it keeps its
  own simpler correctness check (exact string match, no manual corrections),
  same as the private repo's original.
- `collect_chained_versions.py` handles `chained_prompting_rosetta.py`'s
  filename inconsistency: unlike `_pattern.py`/`_monolingual.py`, it omits
  the format name from its output filename. Both naming variants are
  matched automatically.
- Both scripts have only been verified against synthetic multi-run data
  (majority voting, tie-breaking, and matchup pairing extraction all
  checked by hand) — not against a real 32-run API dataset, since producing
  one requires the API cost mentioned above.

## Pre-scored results (`data/results/`)

Beyond the raw model outputs in `responses_obf.zip` / `chained_responses.zip`,
`data/results/` ships the actual scored results tables the paper's headline
numbers are drawn from — the output of the `code/judging/` pipeline above,
already run once so you don't have to. Each `.zip` is password-protected
with `olympiad` (see **Contamination protection**) and, inside, follows the
same layout:

```
<family>.zip
└── <family>/
    ├── deepseek/
    │   ├── <primary result file, n=32>
    │   └── subsamples/
    │       └── <the same result at n=1,2,4,8,16>
    ├── gemini/    (same)
    └── llama/     (same)
```

| Zip | Contents | Exception to the model-triplet layout |
|---|---|---|
| `baseline_shuffle.zip` | `evaluate_subquestions.py` / `subsample_eval.py` output on the shuffled baseline, judge-joined where available | — |
| `chained_prompting.zip` | `evaluate_chained_subquestions.py` output (Rosetta/Pattern/Monolingual combined) | — |
| `matchup.zip` | `evaluate_matchup_multi_run.py` output (Tree-of-Matches) | — |
| `ablation_minimal_prompt.zip` | the §Appendix B reduced-context-prompt ablation | **DeepSeek R1 only** — the paper only ran this ablation on one model, so there's no `deepseek/gemini/llama` split here, and its largest available run is n=16 (no n=32 was recorded for this ablation) |

Two deduplication decisions were made when assembling this from the private
repo's raw output, both verified rather than assumed:

- **`_fix` / judge-joined duplicates**: `baseline_shuffle`'s `n=2,4,8,16,32`
  files are the *judge-joined* versions (`..._fix_with_judge_evaluation.csv`)
  rather than the plain `..._fix.csv` files sitting alongside them in the
  private repo — I diffed a pair by hand and confirmed the judge-joined file
  is a strict superset (same 1,005 rows, every shared column byte-identical,
  10 extra judge columns). `n=1` has no judge-joined counterpart (judging
  needs ≥2 unique answers to choose between), so that one file is the plain
  version.
- **Matchup re-run duplicates**: several matchup sizes had two files with
  different timestamps and genuinely different content (verified by diff,
  not just filename). The latest timestamp was taken as canonical per size;
  one clearly-aborted run (`deepseek_matchup_v29_...`, not a power of two)
  was dropped entirely. Timestamps were stripped from the kept filenames
  since only one now exists per size.

Also dropped as redundant: `llama`'s standalone per-format chained files
(`chained_subquestion_eval_llama_{monolingual,rosetta}_*.csv`), superseded
by the combined `all_formats` files that already include every format, and
a duplicate copy of the size-16 judge-joined file that existed both in
`llama`'s own folder and in the shared judge-output location.

Nothing in `subsamples/` is a strict subset of the `n=32` file above it in
a way that makes it redundant, despite the nested-sampling design (each
smaller subsample's versions are a random subset of the larger one) — each
size's majority vote, tiebreaker, and correctness are recomputed
independently, since that recomputation *is* the inference-time-budget
result being measured. Regenerating them from the `n=32` file with
`subsample_eval.py` will produce a similar but not byte-identical result,
since it doesn't reuse the private repo's exact original random draws.

### Reconstructing Table 2's judge accuracy columns from `baseline_shuffle.zip`

`llm_judge_responses.py` / `llm_mc_responses.py` only judge subquestions with
2+ unique valid answers (`--min_unique_answers 2`, the default) — anything
unanimous is left un-judged, since there's nothing for a judge to choose
between. `join_judge_evaluation.py` reflects this: for a judged row,
`is_judge_correct` / `is_mc_judge_correct` hold the judge's actual correctness
(`True`/`False`); for an un-judged row, they're blank (`is_judge_used` /
`is_mc_judge_used` is `False`), rather than `False`.

Averaging `is_judge_correct` directly (e.g. `df["is_judge_correct"].mean()`
with blanks treated as missing/skipped) therefore computes accuracy **only
over the harder, judged subset** — not the "Rerank judge" / "Top-1 judge"
percentage reported in Table 2, which is measured over *all* 1,005
subquestions. To reproduce the paper's numbers, fall back to the
(un-tied, since un-judged implies <2 unique answers) majority answer for
every row the judge never saw:

```python
df["is_judge_correct_full"] = df.apply(
    lambda r: r["is_judge_correct"] if r["is_judge_used"] else r["is_tiebreaker_correct"],
    axis=1,
)
accuracy_pct = df["is_judge_correct_full"].mean() * 100
```

(substitute `is_mc_judge_correct` / `is_mc_judge_used` for the "Top-1 judge"
column). This reproduces the paper's baseline (29.9) and upper-bound (58.3)
figures exactly when applied to `deepseek/subquestion_eval_deepseek_shuffle_32_fix_with_judge_evaluation.csv`,
and lands within roughly 1 percentage point of the reported Rerank/Top-1
judge figures — the small remaining gap is most likely an artifact of using
the shuffle-split file (whose baseline necessarily differs slightly from the
paper's own, see §3.1/§4.1 of the paper) rather than an error in this
formula. For an exact match to a specific paper figure, re-run
`join_judge_evaluation.py` end-to-end rather than reconstructing it from the
CSV's columns by hand.

## Model configuration

`data/model_list.json` defines every runnable model: its API/local backend
(`model_type`: `openai`, `anthropic`, `cohere`, `open_router`, `transformers`,
`guidance`, `ollama`, ...), any chat template header/footer, and quantization
settings for local models. Add a new model by adding an entry here.

## Known limitations

- `create_benchmark_same_obf.py` needs `data/splits/benchmark.jsonl.zip`,
  which is included, but its own upstream source (the raw, non-obfuscated
  exam corpus) is not part of this repository.
- No generator scripts are included for
  `benchmark_same_obf_{pattern,monolingual,matchup}.jsonl.zip`,
  `benchmark_same_obf_matchup_last_few.jsonl.zip`, or
  `benchmark_{no_matchup,rosetta}.jsonl.zip` — these ship as pre-built
  files only.
- `benchmark_model.py` / `benchmark_model_probabilities.py` default
  `--test_data_zip` to `benchmark.jsonl.zip` (the raw, non-deduplicated
  obfuscation pool). Pass `--test_data_zip` explicitly to point at
  `benchmark_same_obf.jsonl.zip` or another split.
- `code/judging/` has not been run against a real 32-sample API run
  end-to-end (only tested with synthetic data and mocked model calls).
  `evaluate_subquestions.py` and `subsample_eval.py`'s scoring logic
  (in `eval_common.py`) is a port of the private repo's scripts — file
  paths and I/O changed, and one bug was deliberately fixed (see below).
  `evaluate_baseline.py` is a from-scratch rewrite of the same scoring
  logic that skips the `openrouter_runs/` file-splitting step; it was
  checked to produce identical output to `evaluate_subquestions.py` on
  synthetic data, but isn't the literal original code. `llm_judge_responses.py` /
  `llm_mc_responses.py` / `join_judge_evaluation.py` are ports with the
  prompt/parsing logic kept faithful and only paths and the
  `format_fixed` → `format` column renamed. `evaluate_chained_subquestions.py`
  and `evaluate_matchup_multi_run.py` (step 6) are the same story — ported
  and adapted, verified against synthetic multi-run data, not a real 32-run
  API dataset.
- **Random-tiebreak bug fixed during porting**: the private repo's
  `calculate_majority_and_tiebreaker()` called `random.seed(42)` on every
  invocation (i.e. once per subquestion, potentially thousands of times per
  script run) right before its one `random.choice()` call. Because
  `random.choice()` on a freshly-reseeded RNG returns the same index for
  a given tie size every time, every 2-way tie across the entire dataset
  resolved to the same list position (verified: always index 0) rather than
  actually varying — silently biasing the `tiebreaker` metric, and, for a
  subsample of size 2, making it non-representative of true random
  tie-breaking (in the worst case, could make a size-2 subsample's
  tiebreaker accuracy collapse toward a size-1 result rather than
  reflecting genuine coin-flip variance). This repo's version seeds once,
  at the top of each script's entry point, instead — full-run
  reproducibility is preserved (same seed → same output), but each tie now
  gets an independent draw from the advancing random stream. Fixed in
  `eval_common.py` (shared by `evaluate_baseline.py`,
  `evaluate_subquestions.py`, `subsample_eval.py`,
  `evaluate_chained_subquestions.py`) and independently in
  `evaluate_matchup_multi_run.py`'s own copy of the same logic.
- Some duplication remains across `benchmark_model*.py`, `load_questions*.py`,
  and `chained_prompting_{monolingual,pattern,rosetta}.py` — these are
  planned for consolidation into parameterized single scripts in a future
  pass.
