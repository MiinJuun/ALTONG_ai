# Filtering model training

This directory contains offline dataset preparation, validation, model training,
and evaluation tools for the real-time filtering model.

Application runtime code remains in `src/filtering`. Training code may import
the shared schemas, prompts, and policy from that package, but runtime code must
not import from `filtering_training`.

## Setup

Install the CUDA-compatible PyTorch build as described in the repository root
README, then install the training dependencies:

```powershell
python -m pip install -r filtering_training/requirements.txt
```

## Current checks

Run these commands from the repository root:

```powershell
python -m filtering_training.validate_dataset
python -m filtering_training.demo_policy
python -m filtering_training.smoke_test_model
```

Generated checkpoints, adapters, and experiment outputs belong under
`filtering_training/outputs` and must not be committed.

## One-sample model check

```powershell
python -m filtering_training.infer_sample --sample-index 0
```

This runs `Qwen/Qwen3-0.6B` with thinking disabled, validates the four-field
label JSON, and applies the existing policy. The base model has not been
fine-tuned, so this command checks the input/output path rather than accuracy.

## Prepare SFT data

```powershell
python -m filtering_training.prepare_dataset
```

The command validates the samples and writes train, validation, and test JSONL
files plus a split manifest to `filtering_training/outputs/prepared`. The
current synthetic examples are only for pipeline checks. Notifications with the
same text and different contexts stay in one split. The assistant target contains
only the four model label fields; the prompt includes the normalized current
context. The three splits are not a reliable quality benchmark yet.

## Baseline evaluation and LoRA smoke run

```powershell
python -m filtering_training.evaluate --split test
python -m filtering_training.train --max-steps 2
python -m filtering_training.evaluate --split test --adapter filtering_training/outputs/lora-smoke/adapter
```

Run `prepare_dataset` first. Evaluation writes aggregate metrics and run metadata
to `filtering_training/outputs/evaluation`; it does not save raw notifications.
Model responses are not saved unless `--examples-output` is provided. By default, evaluation prints three model JSON examples; use
`--examples-output PATH` to save those examples as UTF-8 JSON. Record reviewed
experiment results and 3-5 examples in `docs/filtering-experiment-log.md`.
The trainer saves a LoRA adapter and run settings under
`filtering_training/outputs/lora-smoke`. It trains only on the assistant JSON
completion and verifies that prompt tokens are masked from the loss. On GPUs that
support it, the trainer uses BF16; otherwise it uses FP16.

The original 34-sample test split had three synthetic examples, including one urgent
notification. Its saved base/adapter reports belong to that dataset snapshot. Run
the baseline again after changing the dataset or split. These numbers cannot establish
model quality. The next quality step is a larger, independently reviewed dataset.

## Synthetic data audit

```powershell
python -m filtering_training.audit_dataset
```

This writes aggregate label and context coverage to
`filtering_training/outputs/audit/dataset_audit.json`. Review the aggregate findings before expanding the dataset.
