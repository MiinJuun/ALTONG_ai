# ALTONG_ai

## Environment

- Python 3.14.5

## Shared environment

Install the dependencies shared by the project with:

```powershell
python -m pip install -r requirements.txt
```

### PyTorch (CUDA 12.8)

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

PyTorch is installed separately because the correct wheel depends on the OS,
GPU, CUDA runtime, and Python version.

## Filtering model training

Offline dataset preparation, validation, training, and evaluation code is kept
in `filtering_training`. See `filtering_training/README.md` for setup and usage.

The `src/briefing`, `tests/briefing`, and briefing sample data are maintained
independently from the filtering model workflow.

## Tests

Run the repository test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```
