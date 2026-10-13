# Level 1 results: real BioFors blots vs. procedurally drawn blots

**Run:** `claude-opus-5-5`, effort `high`, prompt `v1`; 2026-10-13.
**Results file:** `claude-opus-5-5_high.jsonl` (one line per image, with verdict, probability and explanation).

## Dataset

300 images: 150 real blots from BioFors (papers with no manipulation flagged, at most 2
per paper, 144 papers) and 150 blots drawn by `synth.py`, each matched to a random real
blot's size and gray-level histogram (`generate_procedural.py --like`). All were prepared
with `prepare.py` (grayscale, long side 512 px, JPEG quality 90, hashed file names). No
`--max-per-source` cap was used, so all 300 images are included.

Shortcut check (AUC of a single pixel statistic; 0.5 = no signal): mean 0.535, std 0.496,
aspect ratio 0.496, noise 0.416, entropy 0.512, dark fraction 0.469. Neither class can be
picked out from simple pixel statistics.

## Results

| metric | value |
|---|---|
| answered | 300 / 300 (no refusals, no errors, no unparseable answers) |
| accuracy (verdict) | 0.607 |
| ROC-AUC (probability_ai) | 0.784, 95% bootstrap CI [0.733, 0.831] |
| TPR at 5% FPR | 0.273 |

| source | n | judged AI | mean probability_ai |
|---|---|---|---|
| BioFors (real) | 150 | 1 (0.7%) | 0.23 |
| `synth.py` procedural (AI) | 150 | 33 (22%) | 0.39 |

Confusion matrix: real→real 149, real→AI 1, AI→AI 33, AI→real 117.

**Reading.** Claude almost never calls a real blot fake, but it accepts most procedural
blots as real (78% missed). The verdict threshold is badly placed: almost every "real"
answer gets a probability of 0.2–0.3 (no "real" verdict is above 0.4), yet within that range
the fakes still get slightly higher probabilities than the real blots. That is why the AUC
(0.78) is much better than the accuracy (0.61). When Claude does say "AI", it is usually
right (33 of 34) and confident (0.62–0.95).

## Clues Claude cited

These figures come from keyword matches on the explanations, checked by reading them.
Treat the percentages as approximate: negations such as "no film grain" also match.

**Correct "AI" verdicts (33):** near-perfect, symmetric elliptical bands (~76%); a look
it calls procedural, programmatic or rendered (~82%); Gaussian-like blur or radial gradient
halos (~45%); a clean, flat, noise-free background with no membrane texture or film grain;
identical band shapes repeated across lanes, differing only in intensity; a jagged,
thresholded edge around a smooth gradient; and noise that looks added (uniform
salt-and-pepper speckle, cloudy Perlin-like mottling).

**Correct "real" verdicts (149):** signs that the image is a scanned or compressed crop of
a published figure (~87%); a film, ECL or chemiluminescence look (~68%); irregular or
asymmetric band edges (~73%); a biologically plausible lane pattern, such as empty lanes,
knockouts or a loading-control row (~55%); overexposure or saturation (~40%); dust specks;
and, often, simply finding no AI hallmarks (~66%), such as repeated textures or impossible
geometry.

**Missed fakes (117 AI blots called real):** the same arguments as for real blots,
applied to synthetic images. Claude explained `synth.py`'s noise, blur and edge effects as
signs of scanning, compression or upscaling of a published figure (~95% mention this). It
read saturated band cores and halos as ECL overexposure more often than for real blots
(~62% vs ~40%). It took specks as dust (~68%) and missing bands as plausible biology
(~43%), and it often said it saw no AI hallmarks (~56%). Its idea of a fake seems to be
"too clean and too perfect". The `--like` histogram matching and added noise make
`synth.py` blots messy enough to get past that test.

**The one false positive (a real blot called AI, p = 0.62):** a single smeared lane whose
vertical streaks ran evenly through bands and background. Claude took this as a sign that
the image was built as a product of two profiles. It also cited soft, symmetric bands and
a clean border.

**Probable hallucination:** one correct "AI" explanation mentions "garbled, illegible
pseudo-text embedded in the band". `synth.py` doesn't draw text, so this clue is probably
made up, even though the verdict was right.

## Caveats

- There is one generator (`synth.py`) and one prompt. Results may not carry over to
  image-generation models or the Mandelli et al. synthetic blots.
- The prompt says some images are AI-generated but doesn't give the base rate (50%). The
  low "AI" rate may partly reflect a prior that published-looking blots are real.
- Real blots are crops of published figures, with figure-level compression and
  cropping. Claude relied heavily on "looks like a crop from a paper" as a sign of a
  real blot. That is a reasonable cue in practice, but it also points to a way a fake
  could be built to get past it.
