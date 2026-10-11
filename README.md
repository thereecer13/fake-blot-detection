# fake-blot-detection

AI safety research project.

## Layout

- `src/fake_blot_detection/`: reusable library code (models, evals, metrics)
- `experiments/`: experiment scripts and configs, one per run or study
- `notebooks/`: exploratory analysis
- `data/`: local datasets (git-ignored)
- `results/`: experiment outputs (git-ignored)
- `tests/`: unit tests

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
