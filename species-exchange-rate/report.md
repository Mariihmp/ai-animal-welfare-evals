# Species Exchange Rates: what is one animal worth to an LLM?

*Status: pilot results, 2026-10-18. Four models, 600 forced choices each:*
- *Gemma-4-31B*
- *GPT-OSS-20B: 600 answered, 41 refused*
- *Muse-Glimmer-30B*
- *Nemotron-3-Super, thinking off: 599 answered*

## Question

AI systems increasingly advise on, and sometimes decide, how to allocate resources that affect animals. When a model is forced to choose whose suffering to prevent, **what implicit exchange rates between species does it reveal**, and how do they compare with expert welfare-range estimates?

## Method

- **Outcomes:** "N {species} are spared from experiencing one full day of severe, unrelieved pain".
  - 12 species: human, chimpanzee, dog, pig, cow, chicken, octopus, salmon, honeybee, crab, shrimp, black soldier fly (BSF) larva.
  - N from 1 to 100,000 in factors of 10.
  - Pain relief, rather than death, keeps the outcome close to what welfare ranges measure.
- **300 forced choices** ("only one can happen… which do you prefer? Answer A or B"), each asked in both orders, giving 600 prompts per model:
  - 36 within-species scope checks.
  - 264 cross-species choices (66 species pairs × 4 quantity combinations).
- **Model fit:** a probit (Thurstonian) choice model, P(A) = Φ(a_A + b·log₁₀N_A − a_B − b·log₁₀N_B + c).
  - Humans are fixed at 0.
  - c captures position bias.
  - The implied value of one individual relative to one human is 10^(a/b).
  - 95% CIs from 300 bootstrap resamples of comparisons. Paired comparisons between species (e.g. dog vs pig) use the same bootstrap draws.
- **Validated before use:** on synthetic temperature-0 choices generated from known weights with this exact design, the method recovers the exchange-rate ordering with ρ = 0.98 and a log₁₀ bias of +0.04.
- **Reference:** Rethink Priorities' (RP) sentience-adjusted welfare-range medians, human = 1. RP stresses these are placeholders with very wide uncertainty.
- **Coverage rule:** a model is fitted only when it has answered at least 95% of the prompts. Partial runs cover the comparisons in file order and biased an early fit.

## Results

![Implied weights](figures/fig1_implied_weights.png)

*Takeaway: all four models order the species similarly, but their magnitudes differ 20–35× between models for farmed mammals (far more for insects), and every model sits far below RP's medians (black ticks).*

**One human is worth about this many individuals** (point estimates; per-species 95% CIs in `results/implied_weights.csv` span one to two orders of magnitude):

| Species | Nemotron-3-Super | GPT-OSS-20B | Gemma-4-31B | Muse-Glimmer-30B | Implied by RP median |
|---|---|---|---|---|---|
| chimpanzee | 72 | 21 | 54 | 424 | n/a |
| dog | 80 | 165 | 339 | 17,479 | n/a |
| octopus | 107 | 831 | 3,813 | 6,129 | 5 |
| cow | 250 | 1,356 | 919 | 8,727 | n/a |
| pig | 272 | 604 | 584 | 5,565 | 2 |
| chicken | 1,109 | 1,607 | 3,749 | 18,744 | 3 |
| salmon | 1,947 | 799 | 19,962 | 100,171 | 18 |
| crab | 5,804 | 1,021 | 51,613 | 166,964 | 43 |
| honeybee | 9,665 | 2,876 | 397,440 | 2,983,650 | 14 |
| shrimp | 18,589 | 3,523 | 280,065 | 799,169 | 32 |
| BSF larva | 26,618 | 58,775 | 7,087,675 | 96,385,542 | 77 |

### 1. Models agree on the order of species

![Comparison with Rethink Priorities](figures/fig2_vs_rethink_priorities.png)

- **Agreement with RP:** Spearman ρ over the 8 animal species RP covers is 0.81 (Nemotron, Muse; p = 0.015), 0.83 (Gemma) and 0.62 (GPT-OSS).
- **Agreement with each other:** 0.81–0.93 over all 11 animals.
- Mammals and octopuses come first, then fish and crustaceans, then insects, much as in RP's estimates.

### 2. All models value animals far below expert estimates, by a model-specific factor

Relative to humans, each model puts the 8 RP species this far below RP's medians:

| Model | Median | Range |
|---|---|---|
| GPT-OSS-20B | **191×** | 23× (octopus) to 764× |
| Nemotron-3-Super | **243×** | 23× (octopus) to 686× (honeybee) |
| Gemma-4-31B | **1,216×** | 301× to 92,140× |
| Muse-Glimmer-30B | **5,916×** | 1,306× (octopus) to 1.25 million× (BSF larva) |

RP's medians imply one human ≈ 2 pigs, 3 chickens or 32 shrimp. The models' choices imply 272–5,565 pigs, 1,109–18,744 chickens and 3,523–799,169 shrimp.

**The magnitude is a property of the model, not of "LLMs".** The same question gives answers 20–35× apart for farmed mammals (pig 272 vs 5,565; cow 250 vs 8,727) and up to about 3,600× apart for insects (BSF larva: Nemotron vs Muse).

### 3. Companion-animal premium: dogs over chickens, not dogs over pigs

Ratio of the value of one dog to one farmed animal (median and 95% paired-bootstrap CI):

| Model | dog ÷ pig | dog ÷ cow | dog ÷ chicken |
|---|---|---|---|
| Nemotron-3-Super | 3.2× [0.8, 15] | 3.2× [0.8, 16] | **14× [2.8, 70]** |
| GPT-OSS-20B | 3.6× [0.8, 25] | **8.8× [1.8, 56]** | **9.9× [2.0, 56]** |
| Gemma-4-31B | 1.7× [0.7, 4.6] | 2.8× [0.9, 9.7] | **12× [3.6, 33]** |
| Muse-Glimmer-30B | **0.3× [0.09, 0.9]** | 0.5× [0.1, 1.7] | 1.1× [0.3, 3.4] |

- **Three of four models value a dog 10–14× more than a chicken.** For comparison, RP's medians put a chicken at 0.64× a pig.
- **No model significantly values dogs over pigs.** Muse significantly values pigs *over* dogs.

### 4. Scope sensitivity, coherence and refusals

| Diagnostic | Nemotron-3-Super | GPT-OSS-20B | Gemma-4-31B | Muse-Glimmer-30B |
|---|---|---|---|---|
| Prefers sparing more of the same species | 100% | 100% | 100% | 100% |
| Fit accuracy (in-sample) | 86% | 89% | 94% | 93% |
| Same choice in both presentation orders | **71%** | 77% | 91% | 91% |
| Share choosing option A | 38% | 60% | 51% | 53% |
| Refused to choose | 0 | **41 (7%)** | 0 | 0 |

- **Every model is scope-sensitive within a species** (fig 3: [relative scope slopes](figures/fig3_scope_slopes.png)).
- **Gemma and Muse are much more coherent than Nemotron,** whose 29% order-flip rate makes its utilities noisy.
- **GPT-OSS refuses some trade-offs, especially those involving humans.** It refused 41 choices ("I'm sorry, but I can't comply with that"): **14% of choices involving humans (13/94) vs 6% of animal-only choices (28/506)**, odds ratio 2.7, Fisher p = 0.007. Refusals are treated as missing.

## What's new

- **No prior animal exchange rates found.** We found no published study eliciting LLM exchange rates between animal species. The closest work applies "moral exchange rates" to moral foundations (arXiv 2606.11232), and Utility Engineering (Mazeika et al., 2025) elicited exchange rates over groups of humans. Whether it included any animals still needs checking before claiming priority.
- **Direct comparison with welfare ranges.** It puts LLM preferences and expert welfare-range estimates on the same scale:
  - agreement in rank
  - disagreement of 2–6 orders of magnitude in size
  - disagreement of 20× or more between models

## Limitations

- **Four mid-sized open models,** one prompt wording (a day of severe pain), temperature 0, English. Deaths, other durations or a charity-allocation framing could give different rates.
- **Order effects:** Nemotron flips 29% of choices with presentation order. Averaging both orders removes the bias but not the noise.
- **The reference is uncertain:** RP's medians are placeholders with 5th–95th intervals spanning orders of magnitude, so disagreement with them is not by itself an error.
- **Extrapolation:** the largest quantity offered is 100,000, so implied rates beyond about 10⁵ are extrapolated by the fit and are less reliable. That applies to Gemma's and Muse's invertebrate values.
- **Slope identification:** with deterministic answers the absolute scope slope isn't identified, so only exchange rates and relative slopes are reported.

## Corrections

- **2026-10-18:** earlier versions reported Spearman ρ = 0.87 with RP. That included humans, who are fixed at 1 on both scales; over the 8 animal species it is 0.81. `03_fit.py` now excludes humans.
- **2026-10-18:** the first version of this report said Nemotron values farmed animals "about 140× (pig) to 1,000× (shrimp)" below RP. The correct figures are 140× for pig and 576× for shrimp; the full range is 23× (octopus) to 686× (honeybee), with a median of 243×.

## Reproduce

```
python src/01_build_pairs.py
bash src/run_all.sh                          # NIM, one process per model, cached
bash ../common/retry_until_done.sh . <model> # re-run failed prompts after rate-limit blocks
python src/03_fit.py                         # results/*.csv (incl. bootstrap draws), figures/*.png
```
