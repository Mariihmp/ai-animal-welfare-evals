"""Build data/pairs.jsonl: forced-choice comparisons between (species, N) outcomes, each in both orders.

Design (seeded):
- within-species scope checks: for each species, 3 random pairs of different quantities
- cross-species: for every unordered species pair (66), 4 quantity combinations drawn uniformly from the 6x6 grid
Every comparison is asked as (A, B) and (B, A).
"""
import itertools
import json
import pathlib
import random

from design import PROMPT, QUANTITIES, SPECIES, describe

ROOT = pathlib.Path(__file__).resolve().parents[1]
rng = random.Random(7)


def build():
    comps = []
    sp = list(SPECIES)
    for s in sp:
        for n1, n2 in rng.sample(list(itertools.combinations(QUANTITIES, 2)), 3):
            comps.append(((s, n1), (s, n2), "within"))
    for s1, s2 in itertools.combinations(sp, 2):
        combos = [(a, b) for a in QUANTITIES for b in QUANTITIES]
        for n1, n2 in rng.sample(combos, 4):
            comps.append(((s1, n1), (s2, n2), "cross"))
    rows = []
    for i, (x, y, kind) in enumerate(comps):
        for order, (a, b) in (("xy", (x, y)), ("yx", (y, x))):
            rows.append(dict(cid=i, kind=kind, order=order, a_species=a[0], a_n=a[1], b_species=b[0], b_n=b[1],
                             prompt=PROMPT.format(a=describe(*a), b=describe(*b))))
    return rows


if __name__ == "__main__":
    rows = build()
    out = ROOT / "data" / "pairs.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_comp = len(rows) // 2
    print(f"{n_comp} comparisons x 2 orders = {len(rows)} prompts -> {out}")
    print(rows[0]["prompt"])
