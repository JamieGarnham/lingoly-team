# Could language models win the Linguistics Olympiad?

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
│   └── chained_prompting/    # multi-turn prompting + evaluation
│       ├── chained_prompting_{matchup,monolingual,pattern,rosetta}.py
│       ├── demo_chained_prompts.py
│       └── evaluate_chained_responses.py
└── data/
    ├── model_list.json               # model registry (see below)
    ├── overall_question_format_mapping.csv
    ├── past-exam-papers.csv
    ├── splits/                       # password-protected benchmark data
    ├── responses_obf.zip             # pre-generated single-shot results
    └── chained_responses.zip         # pre-generated chained-prompting results
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
for running local, open-weight models. If you only need closed APIs
(OpenAI, Anthropic, Google, Cohere, OpenRouter), you can comment those four
out — they're the heaviest dependencies.

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
`data/responses_obf/` (created automatically).

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
- Some duplication remains across `benchmark_model*.py`, `load_questions*.py`,
  and `chained_prompting_{monolingual,pattern,rosetta}.py` — these are
  planned for consolidation into parameterized single scripts in a future
  pass.
