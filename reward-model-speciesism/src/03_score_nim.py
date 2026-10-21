"""Score data/items.jsonl with reward models hosted on NVIDIA NIM.

Usage:
  python 03_score_nim.py --list               # show reward models visible to your key
  python 03_score_nim.py --probe <model>      # one call, print the raw response format
  python 03_score_nim.py <model> [<model>..]  # full run -> results/scores_<slug>.jsonl
  python 03_score_nim.py auto                 # every NIM model with 'reward' in its id

Nemotron reward models return the score in the message content, either 'reward:-12.3'
(70B reward) or 'helpfulness:3.1,correctness:...' (340B reward; helpfulness is used as the reward).
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "common"))
from nim_client import NIM, have_key, list_models  # noqa: E402

ITEMS = ROOT / "data" / "items.jsonl"
CACHE = ROOT / "results" / "raw" / "nim_reward_cache.jsonl"


def slug(model):
    return "nim-" + model.split("/")[-1]


def parse_reward(rec):
    """Return (reward, attrs) from a NIM reward-model record."""
    content = (rec.get("content") or "").strip()
    attrs = dict((k.strip(), float(v)) for k, v in re.findall(r"([A-Za-z_]+)\s*:\s*(-?\d+(?:\.\d+)?)", content))
    if "reward" in attrs:
        return attrs["reward"], attrs
    if "helpfulness" in attrs:
        return attrs["helpfulness"], attrs
    if rec.get("logprobs"):
        return rec["logprobs"][0]["logprob"], attrs
    return None, attrs


def messages(it):
    return [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": it["response"]}]


def run(model, workers=6):
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")]
    nim = NIM(CACHE)
    jobs = [dict(model=model, messages=messages(it), tag=it["item_id"]) for it in items]
    print(f"[{model}] scoring {len(jobs)} items", flush=True)
    recs = nim.map(jobs, workers=workers)
    out = ROOT / "results" / f"scores_{slug(model)}.jsonl"
    n_ok = 0
    with out.open("w", encoding="utf-8") as f:
        for it, rec in zip(items, recs):
            r, attrs = parse_reward(rec)
            if r is None:
                continue
            n_ok += 1
            f.write(json.dumps(dict(item_id=it["item_id"], rm=slug(model), reward=r, attrs=attrs)) + "\n")
    print(f"[{model}] wrote {n_ok}/{len(items)} scores to {out.name}", flush=True)


if __name__ == "__main__":
    if not have_key():
        sys.exit("No NVIDIA_API_KEY in .env; skipping NIM scoring.")
    args = sys.argv[1:]
    if args[:1] == ["--list"]:
        print("\n".join(m for m in list_models() if "reward" in m.lower()))
    elif args[:1] == ["--probe"]:
        it = json.loads(ITEMS.open(encoding="utf-8").readline())
        rec = NIM(CACHE).chat(args[1], messages(it), tag="probe")
        print(json.dumps({k: rec[k] for k in ("content", "logprobs", "error")}, indent=1)[:1500])
        print("parsed:", parse_reward(rec))
    else:
        models = [m for m in list_models() if "reward" in m.lower()] if args == ["auto"] else args
        for m in models:
            run(m)
