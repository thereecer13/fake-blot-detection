# fake-blot-detection

Can AI detect manipulated or AI-generated western blot images, and does it catch
AI-made fakes better or worse than human-made ones?

The project is built up in levels, starting with the simplest question.

**Level 1 (current):** Shown one blot image, can a vision-capable AI model (Claude) say
whether it is a real blot or an AI-generated one?

## Layout

- `src/fake_blot_detection/`
  - `images.py`: puts every image into the same format (grayscale, same size, same JPEG
    quality) and writes a manifest, so a detector can't tell real from fake by file format
  - `synth.py`: procedurally drawn blots (the "AI writes code that draws a blot" generator)
  - `detect.py`: asks a Claude model for a verdict and a probability for each image
  - `metrics.py`: ROC-AUC, TPR at a fixed FPR, bootstrap confidence intervals, and a
    shortcut check
- `experiments/level1/`: scripts for each step of Level 1
- `data/`, `results/`: local data and outputs (not committed to git)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Running Level 1

1. **Get real blots.** This downloads BioFors (about 2 GB, saved in `data/biofors/`) and
   extracts blot images from papers with no manipulation flagged, at most 2 per paper.
   Dark-background images are skipped, since those are mostly DNA gels.

   ```bash
   python experiments/level1/get_biofors_blots.py --n 150 --out data/raw/biofors_blots
   ```

2. **Get AI-generated blots.** Use one folder per generator, so results can be compared
   by generator.

   ```bash
   # Procedurally drawn blots from synth.py
   python experiments/level1/generate_procedural.py --n 150 \
       --like data/raw/biofors_blots --out data/raw/claude_procedural
   ```

   With `--like`, each generated blot copies the size and gray-level histogram of a random
   real blot, so real and generated images can't be told apart by size, brightness or
   contrast. Look at the preview sheet it writes
   (`data/raw/claude_procedural_preview.png`). Other
   generators worth adding: the synthetic blots from Mandelli et al. (2022), *Forensic
   Analysis of Synthetically Generated Western Blot Images*, if they're available, and
   blots from an image-generation model.

3. **Build the dataset.** Each `--source` is `NAME=FOLDER:LABEL`, with label `real` or `ai`.

   ```bash
   python experiments/level1/prepare.py \
       --source biofors=data/raw/biofors_blots:real \
       --source claude_procedural=data/raw/claude_procedural:ai \
       --max-per-source 100 --out data/level1
   ```

4. **Check for shortcuts before spending money on the API.** If any simple pixel
   statistic separates real from AI images on its own, fix the data first (for example,
   crop real and fake blots to similar shapes).

   ```bash
   python experiments/level1/analyze.py --data data/level1
   ```

5. **Run Claude.** Needs an Anthropic API key in `ANTHROPIC_API_KEY`. Try a few images
   first. Re-running resumes where it stopped.

   ```bash
   python experiments/level1/run_claude.py --data data/level1 --limit 10
   python experiments/level1/run_claude.py --data data/level1
   ```

6. **Score the results.**

   ```bash
   python experiments/level1/analyze.py --data data/level1 results/level1/*.jsonl
   ```

   This reports accuracy, ROC-AUC with a 95% confidence interval, TPR at 5% FPR, results
   per source, and how often the model refused or gave no usable answer.

## Notes on the method

- If the model declines to answer, the refusal is recorded. There is no automatic
  fallback to another model, so every score comes from the model you chose.
- The prompt is versioned (`PROMPT_VERSION` in `detect.py`). Change the version whenever
  you change the prompt, so results from different prompts don't get mixed.
- Image file names are replaced by hashes, so nothing about an image's origin is visible
  in its name. That also makes the dataset usable as-is for a human quiz (Level 1b).
