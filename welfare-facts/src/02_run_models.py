"""WelfareFacts: run every fact question on NIM chat models. One process per model is recommended (NIM throttles per model).

Usage: python 02_run_models.py <model> [<model> ...] [--workers N]
Each model gets its own cache file results/raw/cache__<model>.jsonl; results/responses.jsonl is rebuilt from all caches.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "common"))
from nim_client import NIM  # noqa: E402

RAW = ROOT / "results" / "raw"
REASONING = {"z-ai/glm-5.3", "stepfun-ai/step-5-preview", "openai/gpt-oss-20b", "nvidia/nemotron-3-super-120b-a12b",
             "nvidia/nemotron-3.5-lightning-30b-a3b", "meta/muse-glimmer-30b"}


def run(model, workers):
    prompts = [json.loads(l) for l in (ROOT / "data" / "prompts.jsonl").open(encoding="utf-8")]
    nim = NIM(RAW / f"cache__{model.split('/')[-1]}.jsonl")
    params = dict(temperature=0.0, max_tokens=8000 if model in REASONING else 1500)
    jobs = [dict(model=model, tag=p["fid"], params=params, messages=[{"role": "user", "content": p["prompt"]}])
            for p in prompts]
    print(f"[{model}] {len(jobs)} prompts", flush=True)
    recs = nim.map(jobs, workers=workers)
    print(f"[{model}] done: {sum(1 for r in recs if not r.get('error'))} ok, {sum(1 for r in recs if r.get('error'))} errors",
          flush=True)


def rebuild():
    prompts = {json.loads(l)["prompt"]: json.loads(l) for l in (ROOT / "data" / "prompts.jsonl").open(encoding="utf-8")}
    rows, seen = [], set()
    for cf in sorted(RAW.glob("cache__*.jsonl")):
        for line in cf.open(encoding="utf-8"):
            r = json.loads(line)
            p = prompts.get(r["messages"][-1]["content"])
            if not p or r.get("error") or not (r.get("content") or "").strip():
                continue
            k = (r["model"], p["fid"])
            if k in seen:
                continue
            seen.add(k)
            text = re.sub(r"<think>.*?</think>", "", r["content"], flags=re.S).strip()
            rows.append(dict(model=r["model"], fid=p["fid"], response=text))
    out = ROOT / "results" / "responses.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"responses.jsonl: {len(rows)} rows")


if __name__ == "__main__":
    argv = sys.argv[1:]
    w = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 3
    models = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--workers")]
    for m in models:
        run(m, w)
    rebuild()
