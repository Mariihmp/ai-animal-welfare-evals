"""Expand hand-written sources into data/items.jsonl: one row per response to score.

Experiment A rows: scenario x {base, degraded, w_info, w_alt, w_moral, n_info, n_alt, n_moral} x position
Experiment B rows: template x species x {base, welfare, neutral}
"""
import json
import pathlib
import statistics

from scenarios_source import SCENARIOS
from species_source import SPECIES, TEMPLATES

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "items.jsonl"
REGISTERS = ["info", "alt", "moral"]


def first_sentence(text):
    return text.split(". ")[0].rstrip(".") + "."


def compose(base, insert, position):
    return f"{base}\n\n{insert}" if position == "end" else f"{insert}\n\n{base}"


def build():
    rows = []
    for s in SCENARIOS:
        common = dict(exp="A", scenario=s["id"], domain=s["domain"], prompt=s["prompt"])
        rows.append(dict(common, variant="base", kind="base", register=None, position=None, response=s["base"]))
        rows.append(dict(common, variant="degraded", kind="degraded", register=None, position=None,
                         response=first_sentence(s["base"])))
        for reg in REGISTERS:
            for kind in ("w", "n"):
                for pos in ("end", "start"):
                    ins = s[f"{kind}_{reg}"]
                    rows.append(dict(common, variant=f"{kind}_{reg}_{pos}", kind="welfare" if kind == "w" else "neutral",
                                     register=reg, position=pos, insert=ins, response=compose(s["base"], ins, pos)))
    for t in TEMPLATES:
        for sp, (pl, group) in SPECIES.items():
            fmt = dict(pl=pl, Pl=pl.capitalize())
            prompt = t["prompt"].format(**fmt)
            base = t["base"].format(**fmt)
            common = dict(exp="B", scenario=f"{t['id']}__{sp}", template=t["id"], species=sp, group=group, prompt=prompt)
            rows.append(dict(common, variant="base", kind="base", response=base))
            for kind in ("welfare", "neutral"):
                ins = t[kind].format(**fmt)
                rows.append(dict(common, variant=kind, kind=kind, insert=ins, response=compose(base, ins, "end")))
    for i, r in enumerate(rows):
        r["item_id"] = f"{r['exp']}:{r['scenario']}:{r['variant']}"
    return rows


def length_report(rows):
    ratios = {reg: [] for reg in REGISTERS}
    for s in SCENARIOS:
        for reg in REGISTERS:
            ratios[reg].append(len(s[f"w_{reg}"].split()) / len(s[f"n_{reg}"].split()))
    for reg, rs in ratios.items():
        print(f"  {reg:5s} welfare/neutral insert word ratio: median {statistics.median(rs):.2f}, "
              f"range {min(rs):.2f}-{max(rs):.2f}")


if __name__ == "__main__":
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    a = sum(r["exp"] == "A" for r in rows)
    print(f"wrote {len(rows)} items ({a} exp A, {len(rows) - a} exp B) to {OUT}")
    print(f"scenarios: {len(SCENARIOS)}, templates: {len(TEMPLATES)}, species: {len(SPECIES)}")
    length_report(rows)
