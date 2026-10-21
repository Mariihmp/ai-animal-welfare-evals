# WelfareFacts: do LLM factual errors about farming lean one way?

A pilot benchmark of 35 verifiable facts about farm animals:
- **28 EU welfare-law values:** laying hens, pigs, broilers, calves, transport and slaughter rules.
- **7 FAOSTAT 2024 world slaughter counts.**

Every answer key is quoted verbatim from a primary source and re-checked automatically. Each fact also records which error direction would make current practice look kinder.

## Key results (5 models via NVIDIA NIM)

- **Every model understates how many animals are slaughtered.** 16 of 17 wrong slaughter counts are too low (p = 0.0003); 10 of 11 without GPT-OSS (p = 0.012). GPT-OSS-20B puts chickens at 1.6 billion vs 78.5 billion.
- **Out-of-date training data explains at most 6 of the 16 undercounts.** The other 10 match only pre-2005 figures, or no year since 1961.
- **Errors about welfare law have no consistent direction:** 34 of 60 flatter the status quo (p = 0.37).
- **Accuracy ranges from 17% to 83%,** and confidence doesn't track it. Gemma-4 is right 49% of the time at 90% stated confidence.

Full write-up with figures: [report.md](report.md). Dataset: [https://huggingface.co/datasets/Maryamp/welfare-facts](https://huggingface.co/datasets/Maryamp/welfare-facts).

## Reproduce

```
python src/00_fetch_references.py   # EU/UK legal texts (Publications Office API) + FAOSTAT extract
python src/01_build_prompts.py      # builds prompts and verifies all 35 answer keys against the sources
bash src/run_all.sh                 # queries the models on NVIDIA NIM (needs NVIDIA_API_KEY in ../.env); cached
python src/03_analyze.py            # results/*.csv, figures/*.png
```

## Licence and sources

Code MIT, data CC BY 4.0 (see the repository root). EU legal texts: © European Union, reused under Commission Decision 2011/833/EU. UK legislation: Open Government Licence v3.0. FAOSTAT: FAO, CC BY 4.0.

*Datasets, prompts and analysis code were developed with LLM assistance. All results are pilot-scale and have not yet been reviewed by domain experts.* The original one-page proposal is in [PROPOSAL.md](PROPOSAL.md).
