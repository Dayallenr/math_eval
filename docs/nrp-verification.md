# Validation

The standard remote NRP workflow has completed a real model evaluation, in
addition to unit tests, published-image checks, and scripted Harbor checks.

## Completed baseline evaluation

| Field | Value |
| --- | --- |
| Evaluation ID | `math-eval-20261104071253-7a384f` |
| Model | `math-eval-models:Qwen3-8B-SFT-no-labels` |
| Source | `Dayallen/WANG_Lab_Baseline_Model` |
| Model source revision | `26519d9bcaaa09e01f53e0046c6811365b1855d4` |
| Full-suite SHA256 | `b43163a56bb3944876ec01c781feedb8dcd4fcea74ca1d44491f17cd8c1ad121` |
| GPU | One NVIDIA L40S, approximately 48 GB VRAM |
| Generation | 823 saved responses |
| Grading | 823 completed, zero remaining infrastructure errors |
| Reports | Complete and uploaded to `shared-pvc` |

| Benchmark | Correct | Accuracy |
| --- | ---: | ---: |
| MATH-500 | 422/500 | 84.4% |
| BeyondAIME | 14/100 | 14% |
| OlymMATH English HARD | 3/100 | 3% |
| HMMT February 2025 | 4/30 | 13.3% |
| HMMT February 2026 | 6/33 | 18.2% |
| AIME 2025 | 6/30 | 20% |
| AIME 2026 | 6/30 | 20% |

Scores use all questions, including missing final answers and truncated
responses. This is the repository's single-response protocol, not a
reproduction of another publication's prompting or sampling.

One MATH-500 trial initially failed because NRP's Kueue admission webhook timed
out before the task pod was created. A targeted Harbor retry recovered that
trial. All 823 saved predictions and the other 822 valid grades were preserved;
GPU generation was not repeated. The previous failed trial and recovery receipt
remain in the run's `recovery-*` directory. Shared-storage file verification was
slow; a live Python process alone should not be reported as a completed retry.

View the final report inside remote tmux:

```bash
evalctl results math-eval-20261104071253-7a384f
```

## Infrastructure and regression checks

- The published GPU runner passed 31 tests in an NRP CPU verification Job,
  including a tiny real Qwen save/load/generation test. No large baseline model
  was needed for that check.
- The CPU task image passed seven live Harbor checks covering correct, wrong,
  missing, integer, and symbolic answers.
- The coordinator test suite discovered 45 tests: 44 passed and the optional
  Torch/Transformers test skipped in the CPU-only coordinator environment. These
  are the recorded pre-cleanup counts; rerun the current suite after changes.
- The user-authenticated workflow passed a fourteen-response scripted end-to-end
  check, including archive validation, saved-response recovery, Harbor grading,
  and report upload. Scripted results are not model benchmark scores.
- The remote host passed tmux-disconnection, permission, pilot-preview, and two
  live saved-answer Harbor checks with rewards 1 and 0.
- The default pod ServiceAccount could not perform the required eight worker API
  operations. The remote host instead authenticated as the existing user via the
  browser callback flow. No administrator roles were installed.

The pinned runner source commit is `b31ae3442a4bbce816ace113cb7da08ef4605a26`.
Image digests remain in `nrp.example.toml`. The current coordinator source is
copied by the remote-host helper, independently of the older GPU image. Image
verification must use the exact source commit used for that image; current
checkout source is not automatically identical to a pinned published image.

Local diagnostic logs are ignored under `.cache/nrp/`; authoritative responses,
trial records, reports, and checkpoint receipts remain on NRP storage. See
[maintenance](maintenance/README.md) for regression and image-verification
commands.
