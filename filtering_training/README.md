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
