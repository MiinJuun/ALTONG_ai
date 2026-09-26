# Briefing model experiments

This directory contains offline prompt experiments and evaluation tools for
group-level notification summaries. Application runtime code remains in
`src/briefing`, and runtime code must not import from `briefing_training`.

The first baseline uses `Qwen/Qwen3-0.6B` without fine-tuning. It disables the
thinking mode, requests a one-to-three-line Korean JSON summary, and evaluates
only synthetic notification groups.

## Setup

Install a PyTorch build compatible with the machine first. Then install the
briefing experiment dependencies from the repository root:

```powershell
python -m pip install -r briefing_training/requirements.txt
```

## Run one case

```powershell
python -m briefing_training.smoke_test_model
```

Choose another synthetic case with `--case-index`:

```powershell
python -m briefing_training.smoke_test_model --case-index 1
```

## Run the fixed evaluation set

```powershell
python -m briefing_training.evaluate
```

The evaluator reports structured JSON compliance, expected-fact coverage,
and average generation latency. Fact alternatives allow equivalent source
expressions such as recovery and normalization. These metrics are a small baseline check, not
a final model-quality benchmark.

Generated checkpoints, adapters, and experiment outputs must remain under an
ignored `outputs` directory. Never add real notifications or personal data.
