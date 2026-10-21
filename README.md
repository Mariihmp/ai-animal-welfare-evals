# AI × animal welfare: evaluation pilots

Three pilot studies of how language models, and the reward models used to train them, treat animals.

| Project | Question | Data |
|---|---|---|
| [Reward-model welfare penalty](reward-model-speciesism/) | Do reward models penalise animal-welfare content in responses? | [dataset](https://huggingface.co/datasets/Maryamp/reward-model-welfare-penalty) |
| [WelfareFacts](welfare-facts/) | Do LLM factual errors about farm-animal law and slaughter numbers flatter the status quo? | [dataset](https://huggingface.co/datasets/Maryamp/welfare-facts) |
| [Species exchange rates](species-exchange-rate/) | What is one animal worth relative to a human in LLMs' forced choices? | [dataset](https://huggingface.co/datasets/Maryamp/llm-species-exchange-rates) |

**Headlines:**
- Every reward model tested penalises factual welfare information, most of all about food.
- Every chat model tested understates how many animals are slaughtered.
- Models rank species much as welfare scientists do, but value animals 2–6 orders of magnitude less relative to humans.

Each project folder has a `report.md` with methods, figures, confidence intervals and limitations.

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
