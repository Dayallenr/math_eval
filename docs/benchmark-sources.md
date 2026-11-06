# Benchmark selection

The suite contains all questions from seven pinned public benchmark releases.
OlymMATH uses English HARD only. Exact revisions and selections are listed below.

| Benchmark | Publisher release | Selection | Questions | Pinned revision |
| --- | --- | --- | ---: | --- |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | test | 500 | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| BeyondAIME | [ByteDance-Seed/BeyondAIME](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME) | test; all problems | 100 | `c705198ae1043810b1e1693bd879250b51a7a523` |
| OlymMATH (en-hard) | [RUC-AIBOX/OlymMATH](https://huggingface.co/datasets/RUC-AIBOX/OlymMATH) | en-hard / test; excludes EASY, Chinese and Lean | 100 | `2c6532ea2cf929ac1c421532af5951553eaee727` |
| HMMT February 2025 | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) | train | 30 | `6fdc4277120810ff75aa22d2d5489b91f7a262a1` |
| HMMT February 2026 | [MathArena/hmmt_feb_2026](https://huggingface.co/datasets/MathArena/hmmt_feb_2026) | train | 33 | `589a53fe152febc0f4f99f3e561007bf8da8e664` |
| AIME 2025 | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) | train; AIME I + II | 30 | `c94da77eb22bbd6439e62a323bec18493a421302` |
| AIME 2026 | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) | train; AIME I + II | 30 | `d61f486805306e6e6b5913504c07763a99e1b112` |

**Total: 823 questions.** The optional pilot has two per benchmark, 14 total.

Each suite manifest records the exact upstream filenames and their SHA-256 fingerprints.
The full suite always validates the expected source row counts before creating tasks.
AIME 2025 and 2026 include both AIME I and II. BeyondAIME has no upstream IDs, so
its pinned source row ordinals (1-based) are recorded. OlymMATH is read only from
`data/OlymMATH-EN-HARD.jsonl`, with IDs checked to reject another difficulty or language.

AIME answers use exact integer comparison in 0–999. BeyondAIME uses exact integer
comparison without that range restriction. Other math answers use the pinned Math-Verify scorer.
Saved answers that hit the generation limit remain in the overall accuracy denominator;
the report records length-limited responses separately. Merely matching the benchmark
selection does not reproduce another evaluation’s sampling, prompting, or scoring protocol.

The old five-benchmark suite remains on NRP for historical runs. Its scores must be
identified by their original suite fingerprint when compared with new runs.
