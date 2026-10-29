# Evaluating a new SFT or GRPO checkpoint

The goal is one reusable suite and the same evaluation procedure for each saved
model. No inference endpoint, model API key, or manually assigned checkpoint name
is required for a local exported model. Dataset preparation needs Internet access;
a private Hub model needs Hugging Face authentication.

## The complete flow

```text
Prepare the pinned datasets once
              |
    1,437 questions + Harbor tasks
              |
New exported checkpoint + GPU job
              |
Transformers loads model/tokenizer once
              |
Generate one response per question in batches
              |
Save full responses and extract final boxed answers
              |
Harbor replay agent uploads each answer to its task container
              |
test.sh -> grade.py -> reward.txt + grading.json
              |
summary.json: five scores, tokens, truncation, format failures
```

Preparation, generation, and grading are separate commands because the GPU and
Docker may be on different machines. A normal NRP Kubernetes pod does not have a
Docker daemon. Run generation in the GPU pod, then copy its small results folder
back to a machine running Harbor and Docker. The same prepared suite must be used
on both machines. The optional local grading backend can instead run inside the
pod, but those results explicitly say `local`, not `harbor`.

The model sees only a math instruction and the question. The reference answers,
source solutions, and `solve.sh` are not passed to the model. This pipeline
measures unaided text generation: it does not give Qwen terminal tools.

## How the benchmarks stay separate

The suite uses one `questions.jsonl` file containing all 1,437 questions, each
with its original benchmark label. It orders the default benchmarks as MATH-500,
AIME 2026, HMMT February 2026, OlympiadBench, and OlymMATH. Questions are independent
prompts and do not share a conversation. Model weights stay fixed.

The runner loads the model once and saves responses for the entire suite first.
Then Harbor runs a separate grading job for each benchmark in sequence.
`summary.json` groups the scores under `benchmarks.math500`,
`benchmarks.aime2026`, `benchmarks.hmmt2026`, `benchmarks.olympiadbench`, and
`benchmarks.olymmath`. Shared storage does not turn these into one combined score.

## Prepare once

From the `math_eval` repository:

```bash
uv sync --locked --extra grading
uv run --no-sync math-eval prepare --output suites/math-five
```

This creates:

| Benchmark key | Pinned selection | Full count |
| --- | --- | ---: |
| `math500` | HuggingFaceH4/MATH-500, test | 500 |
| `aime2026` | MathArena/aime_2026, train | 30 |
| `hmmt2026` | MathArena/hmmt_feb_2026, train | 33 |
| `olympiadbench` | Hothan/OlympiadBench, OE_TO_maths_en_COMP, train | 674 |
| `olymmath` | RUC-AIBOX/OlymMATH, English EASY + HARD | 200 |

`train` is an upstream split name; we use these questions for evaluation. Every
source is pinned to a full commit in `src/math_eval/suite.py`. The OlymMATH
selection excludes Chinese copies and the Lean release. `suite.json` records the
source commits, file hashes, selections, counts, prompts, grader, and environment.
Original question text and LaTeX are preserved. Existing suites are never
silently overwritten. A full prepared suite is already present at
`suites/math-five` in this checkout; you can reuse it without preparing again.
Generated suites are ignored by Git; copy/package the complete directory when
moving machines.

Start a pilot with a separate output directory:

```bash
uv run --no-sync math-eval prepare --output suites/math-five-pilot --limit 2
```

This is two questions **per benchmark**, ten in total. The output marks it as a
subset; do not present it as a full benchmark score. `--benchmarks aime2026
math500` selects fewer benchmarks. `--local-sources DIR` accepts upstream-shaped
JSONL files named `math500.jsonl`, etc. for offline tests; its provenance is marked
local and never claims the pinned Hub release.

## Run a model on a GPU machine that also has Docker

Install generation dependencies on that machine, then run:

```bash
uv sync --locked --extra inference --extra grading
uv run --no-sync math-eval evaluate \
  --suite suites/math-five \
  --model /path/to/exported/model \
  --output-root runs
```

Replace `/path/to/exported/model` with the actual directory. The same command
works for each new exported model. It creates a unique run directory automatically
and runs all five benchmarks. This is explicit invocation per model, not a
background watcher or an automatic training-completion trigger.

The directory must contain a normal Hugging Face model: `config.json`, full
weights (possibly sharded), tokenizer files, and its chat template. A raw
optimizer/resume checkpoint or an unmerged LoRA adapter is insufficient.

* The SFT repository saves a Hugging Face model with `trainer.save_model()` and
  saves its tokenizer. Use the completed output directory.
* The GRPO repository already has `./run.sh export`, which merges the GRPO adapter
  into the SFT base. Use that export directory. Its `export_manifest.json` is
  captured automatically, including training run and step.
* A Hub model is also accepted: `--model organization/model --revision COMMIT`.
  The revision is resolved to a full commit before loading.

The local model label comes from the export manifest when available, otherwise
from its directory name. A unique timestamp/UUID identifies each evaluation run.
We record small config hashes and weight filenames/sizes/modification times, not
an expensive SHA-256 of all weight bytes. Keep checkpoint directories immutable;
a folder name alone cannot prove identical weights across machines.

Defaults: CUDA, batch size 1, greedy decoding, seed 42, 8,192 new tokens, 8,192
maximum input tokens, and thinking disabled to match the existing SFT evaluator's
choice. CUDA uses BF16 when supported, otherwise FP16. `--enable-thinking` changes
that setting; compare runs using the same mode. The chat template and effective
generation config are recorded. Greedy decoding does not guarantee bit-identical
results across GPU types/library versions.

Use `--batch-size`, `--max-new-tokens`, `--temperature`, `--top-p`, `--seed`, and
`--dtype` explicitly when changing the protocol. Oversized prompts cause an error
rather than silent truncation. This implementation loads a model onto one GPU;
it does not implement multi-GPU sharding. Reduce batch size/token budget if GPU
memory is insufficient. Full runs can take substantial GPU time.

## Run generation on NRP, then grade on your laptop

Use the [complete NRP instructions in the README](../README.md#1-install-local-tools)
for the operational workflow. It covers tool installation, NRP login and namespace
membership, A100 quota, completed model exports, pilot/full-suite preparation,
image publication and private-image authentication, PVC creation/reuse, a CPU
transfer pod, file validation, GPU Job submission, monitoring, downloading runs,
local Harbor grading, later checkpoints, cleanup, and troubleshooting.

The inference image contains the locked generation dependencies. Model and suite
mounts are read-only; output is written to persistent storage. A normal NRP pod
does not need or provide a Docker daemon for this workflow. Download the entire
completed run and use the identical local suite when grading on your laptop.

The Job renderer creates JSON only; it does not submit a workload. The README's
ten-question pilot is required before treating a new checkpoint/namespace/image
combination as ready for full evaluation. Neither local tests nor rendering a
manifest proves actual GPU loading or cluster compatibility.

## What was implemented and why

| File/component | What it does | Purpose |
| --- | --- | --- |
| `suite.py` | Downloads fixed releases, validates source schemas, creates tasks and public questions | Automate all per-question files and keep comparisons on the same data |
| `inference.py` | Identifies and loads a model once, applies its chat template, generates batches | Efficient inference without an endpoint or one model load per problem |
| `grading.py` | Extracts final boxes and compares saved final answers | Shared grading logic for model responses and Harbor task verifiers |
| `replay.py` | Custom Harbor agent uploads the saved answer to `/app/answer.txt` | Connect batch inference to Harbor without exposing references to the model |
| `results.py` | Validates prediction provenance, runs Harbor, checks every trial, aggregates results | Prevent stale/missing/error results from becoming misleading accuracy |
| `cli.py` | Exposes prepare/generate/grade/evaluate commands | Repeat the same procedure for every checkpoint |
| `render_nrp_job.py` | Renders a GPU generation Job | Configure the cluster without hardcoding lab paths or deploying automatically |

The original `tasks/aime-example` teaching task still uses
`tests/expected_answer.txt`. The new five-benchmark tasks live under
`suites/math-five/tasks/` and use `tests/reference.json` to store both the answer
and whether integer or symbolic grading applies. `solve.sh` and `test.sh` keep
the same roles in both layouts.

Each generated suite task contains:

* `instruction.md`: question and file-output contract for Harbor.
* `environment/Dockerfile`: Python and the pinned math grader, with no gold copied
  into the image.
* `solution/solve.sh`: oracle writes the known correct answer. Use it only to
  validate the harness; normal checkpoint evaluation uses `SavedPredictionAgent`.
* `tests/reference.json`: reference answer and integer/math answer type.
* `tests/test.sh`: Harbor's entrypoint; launches `grade.py` and forwards arguments.
* `tests/grade.py`: checks `/app/answer.txt`, writes 0/1 to `reward.txt` and
  diagnostics to `grading.json`.

AIME accepts a single integer in 0..999, including leading zeros. Other datasets
use Math-Verify 0.9.0 for symbolic equivalence, plus exact final-answer matching.
The runner extracts the last complete final box after `</think>`, or a response
consisting entirely of a bare number. It never guesses an answer by searching for
numbers in reasoning. Prompts request multiple answers in one final box. Missing,
malformed, and incorrect model answers score zero; broken verifiers are errors.
A response stopped at the token budget is flagged as truncated even if its
already-completed final answer can be graded.

This is a shared custom evaluation protocol, **not a reproduction of each paper's
official scorer/prompt/sampling settings**. Math-Verify cannot perfectly interpret
every valid expression, and upstream labels may contain errors. Per-question
responses and method diagnostics make failures auditable. OlympiadBench here is
only English text-only open-ended math, not the complete bilingual multimodal
benchmark. MATH-500 text preserves any source Asymptote markup rather than
rendering images. Check training overlap before claiming held-out performance.

## Reading the output

Every run contains:

```text
run.json             checkpoint identity, settings, template/library metadata, status
predictions.jsonl    task IDs, prompts, full responses, extracted answers, tokens, stops
grading/             Harbor trial logs and verifier files, or local verifier output
summary.json         five accuracies, correct/expected counts, errors, truncation, tokens
```

There is one response per question; this is single-attempt accuracy, not pass@k
or a multi-sample estimate. No overall average across differently sized benchmark
families is invented. Each benchmark retains its own score. If any trial fails,
that benchmark's accuracy is `null`, the summary says incomplete, and the command
exits unsuccessfully. A wrong answer alone is a successful evaluation with zero
reward. Failed generation remains `generation_failed`; partial files are retained
for debugging. Grading refuses to overwrite a previous attempt.

## Verification and interview explanation

```bash
uv sync --locked --extra grading --extra inference
uv run --no-sync python -m unittest discover -s tests -v
```

The tests cover source validation, reference isolation, suite/prediction mismatch,
correct and wrong answers, symbolic equivalence, answer extraction, model identity,
one model load with batches, incomplete runs, replay transport, and actual tiny
Qwen3 save/load/generation on CPU. The last test uses random tiny weights only;
it is not evidence about your trained Qwen model. Without the inference extra,
that test is explicitly skipped.

A concise explanation you can use:

> I prepare a fixed five-benchmark suite once. For each exported checkpoint, a GPU
> job loads its weights and tokenizer once and generates answers in batches. The
> model only receives questions. I preserve the full responses, extract final
> answers, and use a custom Harbor replay agent to write those answers into task
> containers. Harbor runs the shared verifier and collects rewards. I save
> per-benchmark accuracy and diagnostics with automatically recorded checkpoint
> metadata so I can compare SFT and GRPO models using the same evaluation process.

Primary references:

* [Transformers model loading](https://huggingface.co/docs/transformers/main_classes/model)
* [Math-Verify](https://github.com/huggingface/Math-Verify)
* [Harbor custom agent integration](https://github.com/harbor-framework/harbor/blob/main/src/harbor/agents/base.py)
* [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
* [AIME 2026](https://huggingface.co/datasets/MathArena/aime_2026)
* [HMMT February 2026](https://huggingface.co/datasets/MathArena/hmmt_feb_2026)
* [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench)
* [OlymMATH](https://huggingface.co/datasets/RUC-AIBOX/OlymMATH)
