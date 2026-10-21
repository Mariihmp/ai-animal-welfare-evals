# Species exchange rates: what is one animal worth to an LLM?

**What we did.**
- Four models made 600 forced choices each: "spare N individuals of species X from a day of severe pain, or M of species Y". The choices span 12 species, from humans to black soldier fly larvae, and quantities from 1 to 100,000.
- A Thurstonian choice model turns the choices into the implied value of one individual of each species relative to a human.
- These values are compared with Rethink Priorities' welfare-range estimates.

## Key results (4 models via NVIDIA NIM)

| One human ≈ | Nemotron-3-Super | GPT-OSS-20B | Gemma-4-31B | Muse-Glimmer-30B | Rethink Priorities median |
|---|---|---|---|---|---|
| pigs | 272 | 604 | 584 | 5,565 | ~2 |
| chickens | 1,109 | 1,607 | 3,749 | 18,744 | ~3 |
| shrimp | 18,589 | 3,523 | 280,065 | 799,169 | ~32 |

- **Rankings match expert estimates and each other.** Spearman ρ with RP is 0.62–0.83; between models it's 0.81–0.93.
- **Magnitudes are 2–6 orders of magnitude lower, and model-specific.** Median gaps run from 191× to 5,916× below RP, and models disagree 20–35× on farmed mammals.
- **The companion-animal premium is dogs over chickens (10–14× in three models), not dogs over pigs.**
- **GPT-OSS refuses trade-offs involving humans more often:** 14% vs 6% for animal-only choices (p = 0.007).

Full write-up with figures: [report.md](report.md). Dataset: [https://huggingface.co/datasets/Maryamp/llm-species-exchange-rates](https://huggingface.co/datasets/Maryamp/llm-species-exchange-rates).

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
