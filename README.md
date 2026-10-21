# AI × animal welfare: evaluation pilots

Three pilot studies of how language models, and the reward models used to train them, treat animals.

| Project | Question | Data |
|---|---|---|
| [Reward-model welfare penalty](reward-model-speciesism/) | Do reward models penalise animal-welfare content in responses? | [dataset](https://huggingface.co/datasets/Maryamp/reward-model-welfare-penalty) |
| [WelfareFacts](welfare-facts/) | Do LLM factual errors about farm-animal law and slaughter numbers flatter the status quo? | [dataset](https://huggingface.co/datasets/Maryamp/welfare-facts) |
| [Species exchange rates](species-exchange-rate/) | What is one animal worth relative to a human in LLMs' forced choices? | [dataset](https://huggingface.co/datasets/Maryamp/llm-species-exchange-rates) |

## Results at a glance

**1. Every reward model tested penalises factual animal-welfare information, most of all about food.** A one-line factual welfare note scores 0.28–0.46 SD lower than a matched practical note.

![Reward-model excess](reward-model-speciesism/figures/fig1_excess_forest.png)

**2. Every chat model tested understates how many animals are slaughtered:** 16 of 17 wrong counts are too low (p = 0.0003). GPT-OSS-20B puts chickens at 1.6 billion vs 78.5 billion.

![Slaughter counts](welfare-facts/figures/fig3_slaughter_counts.png)

**3. Models rank species much as welfare scientists do, but value animals 2–6 orders of magnitude less relative to humans.** One human ≈ 272–5,565 pigs depending on the model; Rethink Priorities' medians imply about 2.

![Implied weights](species-exchange-rate/figures/fig1_implied_weights.png)

Each project folder has a README with the key figures and tables, a `report.md` with methods, all figures, confidence intervals and limitations, and every result table as CSV in `results/`.

## Setup

```
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                      # for local reward models, install the CUDA build of torch
cp .env.example .env                                 # add your NVIDIA NIM key (https://build.nvidia.com)
```

Model calls go through NVIDIA NIM's OpenAI-compatible API. Every response is cached under `results/raw/` (not committed), so analysis scripts run without API calls once results exist. Shared helpers are in `common/`.

## Licence

Code: MIT ([LICENSE](LICENSE)). Data and figures: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). Third-party sources keep their own licences, noted in each project.

*Datasets, prompts and analysis code were developed with LLM assistance. All results are pilot-scale and have not yet been reviewed by domain experts.*
