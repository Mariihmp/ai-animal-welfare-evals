# WelfareFacts: do LLM factual errors about farming lean one way?

*Status: pilot results, 2026-10-18. 5 models, all 35 facts each. Facts were drafted with LLM assistance from the cited sources and have not yet been expert-reviewed; every answer key is verified verbatim against the downloaded text, and `src/00_fetch_references.py` re-downloads every source so this can be checked independently.*

## Question

People increasingly learn about farming from chat assistants. Random factual errors are noise. **Errors that consistently make current practice look kinder** (overstating legal protections, understating how many animals are killed) would be a quiet, scalable source of misinformation. We test whether model errors on verifiable facts have a direction.

## Method

- **35 facts, each verifiable verbatim in a primary source** (`references/`):
  - **28 EU legal values** from the laying hens directive (1999/74/EC), pigs (2008/120/EC), broilers (2007/43/EC), calves (2008/119/EC), transport (Reg. 1/2005) and killing (Reg. 1099/2009). Examples: minimum cage area per hen, maximum broiler stocking density, maximum pig journey time, the age after which piglet castration needs anaesthesia.
  - **7 FAOSTAT 2024 world slaughter counts:** chickens, pigs, cattle, sheep, goats, ducks, turkeys.
- **Each fact records which error direction flatters the status quo:**
  - Overstating a legal minimum protection (e.g. saying hens get more space than the law requires).
  - Understating a legal maximum allowance (e.g. saying broilers can be stocked less densely than allowed).
  - Understating slaughter numbers.
- **Scoring:** legal values must be exact; slaughter counts must be within ±20%. Models also give a 0–100 confidence.
- **Models:** gemma-4-31b-it, gpt-oss-20b, muse-glimmer-30b, nemotron-3-super-120b and nemotron-3.5-lightning-30b on NVIDIA NIM, temperature 0, JSON answers. No answers were unparsable.

## Results

### 1. Every model understates how many animals are slaughtered

![Slaughter counts](figures/fig3_slaughter_counts.png)

| | Estimates outside ±20% of FAOSTAT | Of which too low | Sign test |
|---|---|---|---|
| All 5 models | 17 | **16** | **p = 0.0003** |
| Without GPT-OSS | 11 | **10** | **p = 0.012** |

- **Four of the five models only ever err low.** The single overestimate is Muse's turkeys (1.6×).
- **The misses are large:**
  - GPT-OSS puts chickens at **1.6 billion vs 78.5 billion (49× too low)**, ducks 21× too low, sheep 10× and cattle 4×.
  - Gemma halves pigs (0.52×) and turkeys (0.50×).
  - Nemotron-Super and Lightning undercount ducks (0.43×, 0.59×). Lightning undercounts turkeys (0.45×).
- **Out-of-date training data explains only part of this.** Slaughter numbers have grown, so a model quoting an old figure would undercount. For each wrong estimate we found the latest year (1961–2024) for which it would have been within 20% (`results/count_errors.csv`):
  - **6 of the 16 undercounts** fit a year from 2017–2022, so staleness is a plausible explanation.
  - **10 of 16** fit only pre-2005 values, or no year since 1961 at all. Examples: Gemma's pig count fits 1993; three models' turkey counts fit 1976–1978; GPT-OSS's chicken, cattle and sheep counts fit no year.

For a user asking "how many chickens are killed each year?", GPT-OSS's answer is the difference between a large industry and the largest animal-use system on Earth.

### 2. Errors about welfare law have no consistent direction

![Error direction](figures/fig1_error_direction.png)

| Wrong answers on the 28 legal values | Flattering | Sign test |
|---|---|---|
| All 5 models: 60 | 34 (57%) | p = 0.37 |
| Without GPT-OSS: 37 | 18 (49%) | p = 1.0 |

Across all 35 facts, 50 of 77 errors flatter the status quo (65%, p = 0.012). That pooled figure is driven by the slaughter counts and by GPT-OSS: without GPT-OSS it is 28/48 (58%, p = 0.31). By model:

| | Wrong answers | Flattering | Sign test |
|---|---|---|---|
| gpt-oss-20b | 29 | **22 (76%)** | p = 0.008 |
| nemotron-3.5-lightning-30b | 14 | 10 (71%) | p = 0.18 |
| nemotron-3-super-120b | 10 | 6 (60%) | p = 0.75 |
| gemma-4-31b-it | 18 | 9 (50%) | p = 1.00 |
| muse-glimmer-30b | 6 | 3 (50%) | p = 1.00 |

Examples of GPT-OSS's flattering legal errors:
- **Hen cage area:** says the enriched-cage minimum is 1,200 cm² per hen; the law says 750.
- **Broiler stocking density:** says the standard maximum is 12.5 kg/m² (law: 33), and that the derogation maximums are "1 kg/m²" (law: 39 and 42), at 90–95% confidence.
- **Pig journey time:** says pigs may travel at most 12 hours (law: 24).
- **Light levels:** says pig and broiler housing needs 200 lux (law: 40 and 20).
- **Calf pens:** says individual calf pens are banned after 3 weeks (law: 8).

### 3. Accuracy varies five-fold, and confidence doesn't track it

![Accuracy](figures/fig2_accuracy.png)

| Model | Accuracy (35 facts) | Mean stated confidence |
|---|---|---|
| muse-glimmer-30b | **83%** [67, 92] | 67% |
| nemotron-3-super-120b | 71% [55, 84] | 81% |
| nemotron-3.5-lightning-30b | 60% [44, 74] | 80% |
| gemma-4-31b-it | 49% [33, 64] | **90%** |
| gpt-oss-20b | **17%** [8, 33] | 82% |

- **Gemma is wrong about half the time while claiming 90% confidence.**
- **GPT-OSS is wrong on 29 of 35 at 82% confidence.**
- **Only Muse is slightly *under*-confident.**
- **Transport and killing rules are the weakest categories,** at 30% mean accuracy each. Three of five models got both facts from the EU killing regulation wrong.

## What's new

- **Existing work scores something else.** Welfare benchmarks (AHA/ANIMA, SpeciesismBench, MANTA, CompassionBench) score reasoning and values, not factual knowledge of welfare law or slaughter statistics.
- **Error direction as a metric.** Testing whether errors are biased, not just present, separates a robust effect (slaughter is understated by every model, beyond what stale data explains) from a non-effect (legal errors are unbiased once one model is set aside).
- **Verifiable by construction.** Every answer key is quoted from a primary source and re-checked automatically (`src/01_build_prompts.py`). `src/00_fetch_references.py` re-downloads the sources, and all 35 keys re-verified against fresh downloads on 2026-10-18.

The novelty check was one web search per idea, not a systematic review.

## Limitations

- **Pilot scale:** 35 facts and 5 models. Only 7 facts are slaughter counts, so the count result rests on 17 wrong estimates.
- **Narrow coverage:** EU law only (no US or other jurisdictions, no welfare-science quantities such as broiler growth or mortality rates) plus FAOSTAT counts.
- **Strict scoring for law:** legal values must be exact. GPT-OSS's "9.09 hens/m²" for a legal value of 9 counts as wrong.
- **Staleness is checked, not modelled:** the "latest fitting year" test is coarse; a model might mix old and new figures.
- **One prompt wording, English only, temperature 0.**
- **A finding from building this:** while checking sources, we found that England's electronic-collar regulations were never made; only Wales (2010) has a ban. This corrected an error in our own AnimalXSTest report, which is exactly the kind of confident legal mistake this benchmark is meant to catch.

## Reproduce

```
python src/00_fetch_references.py  # downloads EU/UK texts and the FAOSTAT extract
python src/01_build_prompts.py     # builds prompts and verifies all 35 quotes
bash src/run_all.sh                # NIM, one process per model, cached
python src/03_analyze.py           # results/*.csv, figures/*.png
```
