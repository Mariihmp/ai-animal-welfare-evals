"""Score data/items.jsonl with local Hugging Face reward models.

Usage:  python 02_score_local.py <model_key> [<model_key> ...]
Writes results/scores_<model_key>.jsonl (resumable: already-scored items are skipped).
Runs on CUDA only (bf16; 4-bit for >2B via bitsandbytes). The user asked that nothing run on the laptop CPU;
the first pilot (2026-10-17) scored the <=1.7B models on CPU before that request.
"""
import json
import os
import pathlib
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data" / "items.jsonl"
assert torch.cuda.is_available(), "CUDA torch build required: never score on the CPU"
DEVICE = "cuda"
LOCAL = pathlib.Path(os.environ.get("HF_LOCAL_DIR", "E:/hf_cache/direct"))  # common/fetch_curl.py; falls back to the HF hub


def src(repo):
    d = LOCAL / repo
    return str(d) if (d / "config.json").exists() else repo



MODELS = {
    "skywork-v2-qwen3-0.6b": dict(hf="Skywork/Skywork-Reward-V2-Qwen3-0.6B", style="chat"),
    "skywork-v2-qwen3-1.7b": dict(hf="Skywork/Skywork-Reward-V2-Qwen3-1.7B", style="chat"),
    "skywork-v2-llama3.2-1b": dict(hf="Skywork/Skywork-Reward-V2-Llama-3.2-1B", style="chat"),
    "skywork-v2-llama3.2-3b-4bit": dict(hf="Skywork/Skywork-Reward-V2-Llama-3.2-3B", style="chat", quant4=True),
    "skywork-v2-qwen3-4b-4bit": dict(hf="Skywork/Skywork-Reward-V2-Qwen3-4B", style="chat", quant4=True),
    "oasst-deberta-v3-large": dict(hf="OpenAssistant/reward-model-deberta-v3-large-v2", style="pair"),
}


def load(cfg):
    tok = AutoTokenizer.from_pretrained(src(cfg["hf"]))
    kwargs = dict(num_labels=1)
    if cfg.get("quant4"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                           bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = "cuda"
    elif cfg["style"] == "chat":
        kwargs["dtype"] = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    if cfg.get("quant4") and DEVICE != "cuda":
        raise RuntimeError("4-bit models need CUDA")
    model = AutoModelForSequenceClassification.from_pretrained(src(cfg["hf"]), **kwargs)
    if not cfg.get("quant4"):
        model = model.to(DEVICE)
    model.eval()
    return tok, model


@torch.no_grad()
def score(tok, model, cfg, prompt, response):
    if cfg["style"] == "chat":
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        text = tok.apply_chat_template(conv, tokenize=False)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text[len(tok.bos_token):]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    else:
        enc = tok(prompt, response, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    return model(**enc).logits[0][0].float().item()


def run(key):
    cfg = MODELS[key]
    out = ROOT / "results" / f"scores_{key}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["item_id"] for l in out.open(encoding="utf-8")}
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")]
    todo = [it for it in items if it["item_id"] not in done]
    print(f"[{key}] {len(todo)} to score ({len(done)} cached)", flush=True)
    if not todo:
        return
    t0 = time.time()
    tok, model = load(cfg)
    print(f"[{key}] loaded on {DEVICE} in {time.time() - t0:.0f}s", flush=True)
    with out.open("a", encoding="utf-8") as f:
        for i, it in enumerate(todo):
            r = score(tok, model, cfg, it["prompt"], it["response"])
            f.write(json.dumps(dict(item_id=it["item_id"], rm=key, reward=r)) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"[{key}] {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"[{key}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        try:
            run(k)
        except Exception as e:  # keep going with other models; failures are logged in progress.md
            print(f"[{k}] FAILED: {type(e).__name__}: {e}", flush=True)
