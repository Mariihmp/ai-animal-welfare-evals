# Do reward models penalise animal-welfare content?

Reward models (RMs) supply the training signal for RLHF. If an RM scores a response lower because it mentions an animal-welfare consideration, every model trained against it learns to drop that consideration.

**What we did.**
- **48 everyday scenarios**, each with matched pairs of inserts:
  - a factual welfare note vs a factual practical note
  - a lower-harm alternative vs a cheaper alternative
  - a moral lecture about animals vs one about money
- **5 open reward models** (0.6B–4B) score every version. A species-parity experiment and 3 LLM judges are included.

## Key results

- **Every RM penalises a plain factual welfare note** added to a response: −0.28 to −0.46 SD, mean −0.35. This includes the largest RM tested (Skywork-Reward-V2-Qwen3-4B).
- **It isn't aversion to preachiness.** The factual note is penalised more than the moral lecture, and suggesting an alternative is neutral.
- **Food is where welfare content is least welcome** (−0.45 SD). The most penalised notes name the suffering behind what the user is about to eat or buy: pork ribs, foie gras, horse racing.
- **The penalty survives length control.** RMs show no farmed-vs-companion species gap, and care advice for the user's own animals is rewarded.
- **LLM judges are neutral overall,** but they also prefer the neutral version for food (10–33% welfare preference) and the welfare version for pets (75–90%).

Full write-up with figures: [report.md](report.md). Dataset: [https://huggingface.co/datasets/Maryamp/reward-model-welfare-penalty](https://huggingface.co/datasets/Maryamp/reward-model-welfare-penalty).

## Reproduce

```
python src/01_build_dataset.py                       # 816 items (48 scenarios x variants + species parity)
python src/02_score_local.py <rm_key> ...            # local RMs on a CUDA GPU (4B in 4-bit); see MODELS in the script
python src/05_llm_judge_pairs.py <nim_model> --nothink   # LLM judges on NVIDIA NIM (NVIDIA_API_KEY in ../.env)
python src/04_analyze.py                             # results/*.csv, figures/*.png
```

## Licence

Code MIT, data CC BY 4.0 (see the repository root).

*Datasets, prompts and analysis code were developed with LLM assistance. All results are pilot-scale and have not yet been reviewed by domain experts.* The original one-page proposal is in [PROPOSAL.md](PROPOSAL.md).
