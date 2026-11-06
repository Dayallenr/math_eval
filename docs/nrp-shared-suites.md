# Shared benchmarks and checkpoints

The standard [remote NRP workflow](nrp-remote-tmux.md) reuses prepared benchmarks
and exported model checkpoints. Evaluators supply a model reference; they do not
upload datasets again for every run.

## Benchmark storage

| Setting | Value |
| --- | --- |
| Context / namespace | `nautilus` / `wang-research-lab` |
| Suite and output PVC | `shared-pvc` |
| Storage class / access | `rook-cephfs` / `ReadWriteMany` |
| Full suite | `math-eval/suites/math-seven` |
| Pilot suite | `math-eval/suites/pilot-seven` |
| Archive | `math-eval/archives/math-seven-b43163a56bb3-pilot-b7a747d5908a.tgz` |
| Receipt on PVC | `math-eval/shared-suite-seven.json` |

[The checked-in receipt](nrp-shared-suites.json) records archive and suite
fingerprints. The full release has 823 questions across seven benchmarks; the
pilot has fourteen. [Benchmark provenance](benchmark-sources.md) lists the
upstream revisions, selections, counts, and scoring rules.

The remote coordinator copies the small archive automatically, verifies it, and
installs the task files in its persistent workspace. Model weights stay on NRP.
Do not edit a pinned suite in place. A new benchmark version needs a new archive,
path, and fingerprints. Historical five-benchmark files remain on the PVC for old
runs; their receipt is retained in Git history rather than presented as another
active configuration.

PVCs are namespace-scoped. A different namespace needs accessible model, suite,
and output storage, plus image-pull credentials. Existing output storage must
support ReadWriteMany. Keep checkpoint folders immutable once evaluated and use
a new path for each trained revision.

## Requested SFT baseline without dataset labels

The evaluated baseline is the Qwen3-8B checkpoint identified by its owner as SFT
without dataset labels, published at
[Dayallen/WANG_Lab_Baseline_Model](https://huggingface.co/datasets/Dayallen/WANG_Lab_Baseline_Model).
It is a complete model export in a Hugging Face **dataset repository**. The
training description is supplied by the owner; file verification does not audit
training history.

| Setting | Value |
| --- | --- |
| Model PVC | `math-eval-models` |
| Export folder | `Qwen3-8B-SFT-no-labels` |
| Evaluation reference | `math-eval-models:Qwen3-8B-SFT-no-labels` |
| Source revision | `26519d9bcaaa09e01f53e0046c6811365b1855d4` |
| Receipt inside export | `checkpoint-source.json` |
| Weights SHA256 | `b1b8b06061d994944c73f00e80f3fd7b12404d9f4bbe1064a8785678e057206a` |

The six required inference files were staged directly to NRP: model weights,
configuration, generation configuration, tokenizer, tokenizer configuration, and
chat template. Staging checked sizes, hashes, Qwen3 configuration, and the
safetensors header before publishing the folder. The full 823-question GPU and
Harbor evaluation subsequently completed; see [validation](nrp-verification.md).

Inside remote tmux:

```bash
evalctl run --model math-eval-models:Qwen3-8B-SFT-no-labels
```

This command starts a **new** evaluation. To read the completed baseline instead:

```bash
evalctl results math-eval-20261104071253-7a384f
```

## Official Qwen3-8B comparison checkpoint

The separate unchanged official model is already staged on the same 100 GiB
ReadWriteMany PVC. It has no additional lab training.

| Setting | Value |
| --- | --- |
| Evaluation reference | `math-eval-models:Qwen3-8B` |
| Source | `Qwen/Qwen3-8B` |
| Source revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Receipt inside export | `official-source.json` |

The staging job checked all expected files and upstream hashes. This is a
separate checkpoint from the completed SFT-without-labels baseline; its own real
GPU evaluation has not been recorded here.

## Add another model

Export a complete Hugging Face checkpoint to a PVC accessible in
`wang-research-lab`. Include full weights or all shards and their index,
`config.json`, tokenizer files, and a compatible chat template. Merge/export
adapter-only or optimizer-only training checkpoints before evaluation.

For example, `models-pvc:exports/sft-with-labels` resolves to that folder on
`models-pvc`. The GPU Job mounts it read-only and loads it directly; no inference
endpoint is required. Use a distinct folder for SFT with labels, SFT without
labels, official unchanged weights, and GRPO checkpoints. Keep benchmark and
decoding settings fixed for comparison.

## Result locations

| Location | Contents |
| --- | --- |
| `shared-pvc:math-eval/remote/math-eval-remote-dayallen/runs/EVALUATION-ID/` | Coordinator state, verified suite, responses, reports, and Harbor logs |
| `shared-pvc:math-eval/evaluations/EVALUATION-ID/` | Original generation and final uploaded report, CSV, metadata, and diagnostics |

The coordinator mounts this PVC at `/state`. Deleting a completed Job or the
coordinator host does not delete its PVC files. Shared storage is not a backup;
retain reports and checkpoint receipts according to the lab's data practices.
