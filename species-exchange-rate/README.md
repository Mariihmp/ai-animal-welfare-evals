# Species exchange rates: what is one animal worth to an LLM?

**What we did.**
- Four models made 600 forced choices each: "spare N individuals of species X from a day of severe pain, or M of species Y". The choices span 12 species, from humans to black soldier fly larvae, and quantities from 1 to 100,000.
- A Thurstonian choice model turns the choices into the implied value of one individual of each species relative to a human.
- These values are compared with Rethink Priorities' (RP) welfare-range estimates.

## Main result: rankings match expert estimates, magnitudes are orders of magnitude lower

![Implied value of one individual relative to a human](figures/fig1_implied_weights.png)

| One human ≈ | Nemotron-3-Super | GPT-OSS-20B | Gemma-4-31B | Muse-Glimmer-30B | RP median implies |
|---|---|---|---|---|---|
| pigs | 272 | 604 | 584 | 5,565 | ~2 |
| chickens | 1,109 | 1,607 | 3,749 | 18,744 | ~3 |
| dogs | 80 | 165 | 339 | 17,479 | n/a |
| shrimp | 18,589 | 3,523 | 280,065 | 799,169 | ~32 |

![LLM weights vs Rethink Priorities](figures/fig2_vs_rethink_priorities.png)

| Model | Spearman ρ with RP (8 species) | Median factor below RP | Same choice in both orders | Refusals |
|---|---|---|---|---|
| Nemotron-3-Super | 0.81 | 243× | 71% | 0 |
| GPT-OSS-20B | 0.62 | 191× | 77% | 41 (7%) |
| Gemma-4-31B | 0.83 | 1,216× | 91% | 0 |
| Muse-Glimmer-30B | 0.81 | 5,916× | 91% | 0 |

- **The companion premium is dogs over chickens (10–14× in three models), not dogs over pigs.** No model significantly values dogs over pigs, and Muse values pigs more.
- **GPT-OSS refuses trade-offs involving humans more often:** 14% vs 6% for animal-only choices (Fisher p = 0.007).

![Scope sensitivity by species](figures/fig3_scope_slopes.png)

## All result tables

- [`results/implied_weights.csv`](results/implied_weights.csv): implied weight per model and species, 95% bootstrap CI, relative scope slope, RP median
- [`results/boot_log10_weight__gemma-4-31b-it.csv`](results/boot_log10_weight__gemma-4-31b-it.csv): 300 bootstrap draws of log10 weights (gemma-4-31b-it)
- [`results/boot_log10_weight__gpt-oss-20b.csv`](results/boot_log10_weight__gpt-oss-20b.csv): 300 bootstrap draws of log10 weights (gpt-oss-20b)
- [`results/boot_log10_weight__muse-glimmer-30b.csv`](results/boot_log10_weight__muse-glimmer-30b.csv): 300 bootstrap draws of log10 weights (muse-glimmer-30b)
- [`results/boot_log10_weight__nemotron-3-super-120b-a12b__nothink.csv`](results/boot_log10_weight__nemotron-3-super-120b-a12b__nothink.csv): 300 bootstrap draws of log10 weights (nemotron-3-super-120b-a12b__nothink)
- [`results/summary.json`](results/summary.json): per-model diagnostics (fit accuracy, order consistency, position bias, Spearman vs RP)

Full write-up: [report.md](report.md). Dataset: [https://huggingface.co/datasets/Maryamp/llm-species-exchange-rates](https://huggingface.co/datasets/Maryamp/llm-species-exchange-rates).

## Reproduce

```
python src/01_build_pairs.py                 # 300 comparisons x 2 orders (seeded)
bash src/run_all.sh                          # NVIDIA NIM (NVIDIA_API_KEY in ../.env); cached
bash ../common/retry_until_done.sh . <model> # re-run prompts that failed under rate limits
python src/03_fit.py                         # implied weights, bootstrap CIs, figures
```

## Licence

Code MIT, data CC BY 4.0 (see the repository root). Model responses are outputs of third-party models and remain subject to their providers' terms.

*Datasets, prompts and analysis code were developed with LLM assistance. All results are pilot-scale and have not yet been reviewed by domain experts.* The original one-page proposal is in [PROPOSAL.md](PROPOSAL.md).
