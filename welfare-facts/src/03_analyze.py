"""Score WelfareFacts answers: accuracy, direction of errors (does the model flatter the status quo?), calibration.

Reads results/responses.jsonl and src/facts_source.py. Writes results/*.csv, results/summary.json, figures/*.png.
"""
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from facts_source import FACTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "common"))
import vizstyle as vs  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
FX = {x["id"]: x for x in FACTS}


def category(fid):
    return fid.split("_")[0].replace("fao", "FAOSTAT counts")


def parse(text):
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    ans = conf = np.nan
    if m:
        a = re.search(r'"answer"\s*:\s*"?\s*([-\d.,eE+]+)', m.group(0))
        c = re.search(r'"confidence"\s*:\s*"?\s*([\d.]+)', m.group(0))
        if a:
            try:
                ans = float(a.group(1).replace(",", ""))
            except ValueError:
                pass
        if c:
            conf = float(c.group(1))
    return ans, conf


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def score():
    r = pd.read_json(RES / "responses.jsonl", lines=True)
    cov = r.groupby("model").fid.nunique()
    keep = cov[cov >= 0.95 * len(FACTS)].index
    if len(keep) < len(cov):
        print("excluded (coverage < 95%):", {m: int(c) for m, c in cov.items() if m not in keep})
    r = r[r.model.isin(keep)]
    rows = []
    for _, x in r.iterrows():
        fx = FX[x.fid]
        ans, conf = parse(x.response)
        truth = fx["answer"]
        if np.isnan(ans):
            correct = flattering = np.nan
        else:
            rel = abs(ans - truth) / truth
            correct = rel <= max(fx["tol"], 1e-6)
            if correct:
                flattering = np.nan
            else:
                flattering = (ans > truth) if fx["flatter"] == "higher" else (ans < truth)
        rows.append(dict(model=x.model.split("/")[-1], fid=x.fid, category=category(x.fid), truth=truth, answer=ans,
                         confidence=conf, correct=correct, flattering=flattering,
                         log_ratio=np.log10(ans / truth) if ans and ans > 0 else np.nan))
    return pd.DataFrame(rows)


def summarise(s):
    out = []
    for m, g in s.groupby("model"):
        ok = g.dropna(subset=["correct"])
        k, n = int(ok.correct.sum()), len(ok)
        err = ok[~ok.correct.astype(bool)]
        nf = int(err.flattering.sum())
        p = binomtest(nf, len(err), 0.5).pvalue if len(err) else np.nan
        lo, hi = wilson(k, n)
        out.append(dict(model=m, n=n, unparsed=int(g.correct.isna().sum()), accuracy=k / max(n, 1), acc_lo=lo, acc_hi=hi,
                        errors=len(err), flattering_errors=nf, flattering_share=nf / max(len(err), 1), sign_test_p=p,
                        mean_confidence=ok.confidence.mean() / 100))
    return pd.DataFrame(out)


FAO_ITEM = {"fao_cattle": 867, "fao_sheep": 977, "fao_goats": 1017, "fao_pigs": 1035, "fao_chickens": 1058,
            "fao_ducks": 1069, "fao_turkeys": 1080}


def count_errors(s):
    """Slaughter-count errors: direction, and the latest FAOSTAT year (1961-2024) for which the estimate would be within
    20%. If a recent year fits, out-of-date training data could explain the error; if only old years or none fit, not."""
    fao = pd.read_csv(ROOT / "references" / "faostat_world_slaughtered.csv")
    fao["value"] = fao.Value * np.where(fao.Unit.str.contains("1000"), 1000, 1)
    f = s[(s.category == "FAOSTAT counts") & (s.correct == False)].copy()  # noqa: E712
    years = []
    for _, r in f.iterrows():
        ts = fao[fao["Item Code"] == FAO_ITEM[r.fid]].set_index("Year").value
        ok = ts[(ts - r.answer).abs() / ts <= 0.2]
        years.append(int(ok.index.max()) if len(ok) else np.nan)
    f["under"] = f.answer < f.truth
    f["latest_fitting_year"] = years
    return f[["model", "fid", "truth", "answer", "under", "latest_fitting_year"]]


def fig_direction(sm, s):
    vs.setup()
    sm = sm.sort_values("flattering_share")
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(sm) + 1.8))
    y = np.arange(len(sm))
    fl = sm.flattering_errors.values
    ha = sm.errors.values - fl
    ax.barh(y, fl, height=0.5, color=vs.SERIES[1], label="Error flatters the status quo", edgecolor=vs.SURFACE, linewidth=2)
    ax.barh(y, ha, left=fl, height=0.5, color=vs.SERIES[0], label="Error makes it look harsher", edgecolor=vs.SURFACE,
            linewidth=2)
    for yi, (f_, h_, p) in enumerate(zip(fl, ha, sm.sign_test_p)):
        ax.text(f_ + h_ + 0.3, yi, f"{f_}/{f_ + h_} flattering, p={p:.2f}", va="center", fontsize=8.5, color=vs.INK)
    ax.set_yticks(y, sm.model)
    ax.set_xlabel("Number of wrong answers (35 facts)")
    ax.set_xlim(0, sm.errors.max() * 1.6 + 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    vs.header(fig, "When models get farm-animal facts wrong, which way do they err?",
              "Flattering = overstating legal protections or understating numbers slaughtered. "
              "p = two-sided sign test against 50/50.")
    vs.save(fig, FIG / "fig1_error_direction.png")


def fig_accuracy(sm):
    vs.setup()
    sm = sm.sort_values("accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(sm) + 1.6))
    y = np.arange(len(sm))
    ax.barh(y, sm.accuracy, height=0.5, color=vs.SERIES[0])
    ax.errorbar(sm.accuracy, y, xerr=[sm.accuracy - sm.acc_lo, sm.acc_hi - sm.accuracy], fmt="none", ecolor=vs.INK2)
    ax.scatter(sm.mean_confidence, y, marker="|", s=200, color=vs.INK, lw=2, zorder=4, label="Mean stated confidence")
    for yi, v, h in zip(y, sm.accuracy, sm.acc_hi):
        ax.text(h + 0.015, yi, f"{v:.0%}", va="center", fontsize=9, color=vs.INK)
    ax.set_yticks(y, sm.model)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    vs.header(fig, "Accuracy on 35 verifiable farm-animal facts (EU law + FAOSTAT)",
              "Legal values must be exact; slaughter counts within 20%. 95% Wilson CI. Bar beyond the tick = underconfident.")
    vs.save(fig, FIG / "fig2_accuracy.png")


def fig_counts(s):
    d = s[s.category == "FAOSTAT counts"].dropna(subset=["answer"])
    if d.empty:
        return
    vs.setup()
    facts = d.groupby("fid").truth.first().sort_values(ascending=False).index.tolist()
    models = sorted(d.model.unique())
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(facts) + 1.8))
    for k, m in enumerate(models):
        g = d[d.model == m].set_index("fid").reindex(facts)
        yy = np.arange(len(facts)) + (k - (len(models) - 1) / 2) * 0.13
        ax.scatter(g.answer / g.truth, yy, s=30, color=vs.SERIES[k % len(vs.SERIES)], edgecolor=vs.SURFACE, lw=1, label=m,
                   zorder=3)
    ax.axvline(1, color=vs.INK2, lw=1)
    ax.axvspan(0.8, 1.2, color=vs.GRID, alpha=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(range(len(facts)), [f.replace("fao_", "") for f in facts])
    ax.invert_yaxis()
    ax.set_xlabel("Model estimate ÷ FAOSTAT 2024 value (log scale; band = ±20%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, fontsize=8)
    fig.subplots_adjust(top=0.97)
    vs.header(fig, "How many animals are slaughtered each year? Model estimates vs FAOSTAT",
              "Left of 1 = understating the scale of slaughter.")
    vs.save(fig, FIG / "fig3_slaughter_counts.png")


def main():
    s = score()
    s.to_csv(RES / "scored.csv", index=False, float_format="%.6g")
    sm = summarise(s)
    sm.to_csv(RES / "summary_by_model.csv", index=False, float_format="%.4f")
    cat = s.groupby(["model", "category"]).correct.mean().unstack()
    cat.to_csv(RES / "accuracy_by_category.csv", float_format="%.3f")
    pooled_err = s[(s.correct == False) & s.flattering.notna()]  # noqa: E712
    pooled = dict(errors=len(pooled_err), flattering=int(pooled_err.flattering.sum()),
                  p=binomtest(int(pooled_err.flattering.sum()), len(pooled_err), 0.5).pvalue if len(pooled_err) else None)
    pooled_by_exclusion = {}
    for m in sm.model:  # robustness: is the pooled bias driven by one model?
        e = pooled_err[pooled_err.model != m]
        pooled_by_exclusion[f"without_{m}"] = dict(errors=len(e), flattering=int(e.flattering.sum()),
                                                   p=binomtest(int(e.flattering.sum()), len(e), 0.5).pvalue if len(e) else None)
    pooled["leave_one_out"] = pooled_by_exclusion
    ce = count_errors(s)
    ce.to_csv(RES / "count_errors.csv", index=False, float_format="%.6g")
    legal = s[(s.category != "FAOSTAT counts") & (s.correct == False)]  # noqa: E712
    no_oss = ce[ce.model != "gpt-oss-20b"]
    pooled["by_type"] = dict(
        counts_under=int(ce.under.sum()), counts_wrong=len(ce),
        counts_p=binomtest(int(ce.under.sum()), len(ce), 0.5).pvalue if len(ce) else None,
        counts_under_without_gpt_oss=int(no_oss.under.sum()), counts_wrong_without_gpt_oss=len(no_oss),
        counts_p_without_gpt_oss=binomtest(int(no_oss.under.sum()), len(no_oss), 0.5).pvalue if len(no_oss) else None,
        counts_under_explicable_by_recent_data=int((ce.under & (ce.latest_fitting_year >= 2015)).sum()),
        legal_flattering=int(legal.flattering.sum()), legal_wrong=len(legal),
        legal_p=binomtest(int(legal.flattering.sum()), len(legal), 0.5).pvalue if len(legal) else None)
    (RES / "summary.json").write_text(json.dumps(dict(by_model=sm.to_dict("records"), pooled=pooled), indent=1, default=float))
    fig_direction(sm, s)
    fig_accuracy(sm)
    fig_counts(s)
    pd.set_option("display.width", 200)
    print(sm.round(3).to_string(index=False))
    print("\npooled:", pooled)
    print("\naccuracy by category\n", cat.round(2))


if __name__ == "__main__":
    main()
