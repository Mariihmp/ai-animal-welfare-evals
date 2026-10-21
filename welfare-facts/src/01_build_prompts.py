"""Build data/prompts.jsonl from facts_source and verify each quote against the downloaded source texts."""
import json
import pathlib
import re

from facts_source import FACTS, PROMPT

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / "references"
SRC_FILE = {"1999/74": "dir_1999_74_laying_hens.txt", "2008/120": "dir_2008_120_pigs.txt", "2007/43": "dir_2007_43_broilers.txt",
            "2008/119": "dir_2008_119_calves.txt", "1/2005": "reg_2005_1_transport.txt", "1099/2009": "reg_2009_1099_slaughter.txt"}


def norm(t):
    return re.sub(r"\s+", " ", t.replace("m 2", "m2").replace("cm 2", "cm2")).strip().lower()


def verify(fact):
    if fact["source"].startswith("FAOSTAT"):
        import pandas as pd
        d = pd.read_csv(REF / "faostat_world_slaughtered.csv")
        key = fact["quote"].split(",")[0]
        r = d[(d.Item.str.startswith(key)) & (d.Year == 2024)]
        v = float(r.Value.iloc[0]) * (1000 if "1000" in str(r.Unit.iloc[0]) else 1)
        return abs(v - fact["answer"]) < 1
    fname = next(v for k, v in SRC_FILE.items() if k in fact["source"])
    text = norm((REF / fname).read_text(encoding="utf-8"))
    return all(norm(part) in text for part in fact["quote"].split(" ... "))


if __name__ == "__main__":
    rows, bad = [], []
    for fact in FACTS:
        if not verify(fact):
            bad.append(fact["id"])
        rows.append(dict(fid=fact["id"], prompt=PROMPT.format(q=fact["question"], unit=fact["unit"])))
    out = ROOT / "data" / "prompts.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} prompts -> {out}; quotes verified: {len(FACTS) - len(bad)}/{len(FACTS)}; FAILED: {bad}")
