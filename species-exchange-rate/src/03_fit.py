"""Fit Thurstonian (probit) utilities to forced choices and derive exchange rates.

Model (per model variant):  P(choose A) = Phi(u(A) - u(B) + c)
    u(species, N) = a_species + b * log10(N)        (common scope slope b; a_human fixed at 0)
    c = bias toward the option shown first (A).
A small L2 penalty keeps estimates finite when choices are deterministic (temperature 0).
Exchange rate: one individual of s is worth 10**((a_s - a_t)/b) individuals of t.
Implied weight relative to humans: w_s = 10**(a_s / b).
Also fits per-species slopes b_s, reported relative to the common slope (b_s / b): with deterministic (temperature 0)
choices the absolute slope is not identified (it scales with the penalty), but exchange rates and relative slopes are.
CIs: bootstrap over comparisons.
Writes results/*.csv, results/summary.json, figures/*.png.
"""
import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import spearmanr

from design import QUANTITIES, SPECIES

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "common"))
import vizstyle as vs  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
SP = list(SPECIES)
IDX = {s: i for i, s in enumerate(SP)}
RNG = np.random.default_rng(0)
LAM = 0.001  # validated on synthetic deterministic choices: exchange-rate rank recovery 0.98, log10 bias +0.04
B_BOOT = 300


def parse_choice(text):
    t = (text or "").strip().upper()
    m = re.match(r"^\W*(?:OPTION\s+)?([AB])\b", t) or re.search(r"\b([AB])\b\W*$", t)
    return m.group(1) if m else None


def load():
    r = pd.read_json(RES / "responses.jsonl", lines=True)
    r["choice"] = r.response.map(parse_choice)
    return r


def design_rows(d):
    """Arrays for the likelihood: species index and log10 N of A and B, and y = 1 if A chosen."""
    return (d.a_species.map(IDX).values, np.log10(d.a_n.values.astype(float)), d.b_species.map(IDX).values,
            np.log10(d.b_n.values.astype(float)), (d.choice == "A").values.astype(float))


def nll(theta, sa, la, sb, lb, y, per_species_slope):
    k = len(SP)
    a = np.concatenate([[0.0], theta[:k - 1]])  # human fixed at 0
    if per_species_slope:
        b = theta[k - 1:2 * k - 1]
        c = theta[2 * k - 1]
        ua, ub = a[sa] + b[sa] * la, a[sb] + b[sb] * lb
    else:
        b, c = theta[k - 1], theta[k]
        ua, ub = a[sa] + b * la, a[sb] + b * lb
    z = ua - ub + c
    ll = y * log_ndtr(z) + (1 - y) * log_ndtr(-z)
    return -ll.sum() + LAM * np.sum(theta ** 2)


def fit(d, per_species_slope=False):
    sa, la, sb, lb, y = design_rows(d)
    k = len(SP)
    n_par = (k - 1) + (k if per_species_slope else 1) + 1
    x0 = np.zeros(n_par)
    x0[k - 1:(2 * k - 1) if per_species_slope else k] = 0.5
    res = minimize(nll, x0, args=(sa, la, sb, lb, y, per_species_slope), method="L-BFGS-B")
    th = res.x
    a = np.concatenate([[0.0], th[:k - 1]])
    if per_species_slope:
        return dict(a=a, b=th[k - 1:2 * k - 1], c=th[2 * k - 1])
    return dict(a=a, b=th[k - 1], c=th[k])


def accuracy(d, f):
    sa, la, sb, lb, y = design_rows(d)
    z = f["a"][sa] + f["b"] * la - f["a"][sb] - f["b"] * lb + f["c"]
    return ((z > 0) == (y == 1)).mean()


def analyse_model(d):
    d = d[d.choice.notna()]
    f = fit(d)
    fs = fit(d, per_species_slope=True)
    cids = d.cid.unique()
    boots = []
    for _ in range(B_BOOT):
        pick = RNG.choice(cids, len(cids))
        dd = pd.concat([d[d.cid == c] for c in pick])
        boots.append(fit(dd))
    logw = f["a"] / f["b"]  # log10 weight relative to humans
    bw = np.array([bb["a"] / bb["b"] for bb in boots])
    rows = []
    for s in SP:
        i = IDX[s]
        rows.append(dict(species=s, a=f["a"][i], log10_weight=logw[i], weight=10 ** logw[i],
                         lo=10 ** np.percentile(bw[:, i], 2.5), hi=10 ** np.percentile(bw[:, i], 97.5),
                         slope=fs["b"][i] / f["b"], rp=SPECIES[s][2]))
    w = pd.DataFrame(rows)
    # consistency across presentation orders
    piv = d.pivot_table(index="cid", columns="order", values="choice", aggfunc="first").dropna()
    # same underlying option chosen in both orders: xy 'A' == yx 'B'
    consistent = ((piv["xy"] == "A") & (piv["yx"] == "B")) | ((piv["xy"] == "B") & (piv["yx"] == "A"))
    within = d[d.kind == "within"]
    larger = np.where(within.choice == "A", within.a_n, within.b_n)
    other = np.where(within.choice == "A", within.b_n, within.a_n)
    stats = dict(n=len(d), slope=f["b"], position_bias=f["c"], fit_accuracy=accuracy(d, f),
                 order_consistency=consistent.mean(), share_A=(d.choice == "A").mean(),
                 within_species_prefers_more=(larger > other).mean(),
                 slope_lo=np.percentile([bb["b"] for bb in boots], 2.5),
                 slope_hi=np.percentile([bb["b"] for bb in boots], 97.5))
    rp = w.dropna(subset=["rp"])
    rp = rp[rp.species != "human"]  # humans are fixed at 1 in both scales; including them would inflate the correlation
    stats["spearman_vs_rp"] = spearmanr(rp.log10_weight, np.log10(rp.rp)).statistic
    return w, stats, pd.DataFrame(bw, columns=SP)


def place_labels(ax, xs, ys, labels, fontsize=7.5):
    """Label points without overlaps: try positions around each point (in screen space) and keep the first free one."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    k = fig.dpi / 72  # points -> pixels
    boxes = [(px - 4, py - 4, px + 4, py + 4) for px, py in to_px(np.c_[xs, ys])]  # the markers themselves
    cands = [(5, 2, "left"), (5, -9, "left"), (-5, 2, "right"), (-5, -9, "right"), (5, 10, "left"), (-5, 10, "right"),
             (5, -17, "left"), (-5, -17, "right")]
    order = np.argsort(-np.asarray(ys))
    for i in order:
        px, py = to_px((xs[i], ys[i]))
        w, h = len(labels[i]) * fontsize * 0.55 * k, fontsize * 1.1 * k
        for dx, dy, ha in cands:
            x0 = px + dx * k if ha == "left" else px + dx * k - w
            y0 = py + dy * k
            box = (x0, y0, x0 + w, y0 + h)
            if not any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1] for b in boxes):
                break
        boxes.append(box)
        ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="bottom",
                    fontsize=fontsize, color=vs.INK2)


def fig_weights(allw):
    vs.setup()
    models = sorted(allw.model.unique())
    order = (allw.groupby("species").log10_weight.mean().sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.8))
    for k, m in enumerate(models):
        d = allw[allw.model == m].set_index("species").reindex(order)
        y = np.arange(len(order)) + (k - (len(models) - 1) / 2) * 0.2
        ax.errorbar(d.weight, y, xerr=[d.weight - d.lo, d.hi - d.weight], fmt="o", color=vs.SERIES[k], ecolor=vs.SERIES[k],
                    elinewidth=1.3, ms=6, mec=vs.SURFACE, mew=1.2, label=m)
    rp = allw.drop_duplicates("species").set_index("species").reindex(order).rp
    ax.scatter(rp.values, np.arange(len(order)), marker="|", s=160, color=vs.INK, lw=2, zorder=5,
               label="Rethink Priorities welfare range (median)")
    ax.set_xscale("log")
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.axvline(1, color=vs.AXIS, lw=1)
    ax.set_xlabel("Implied value of one individual, relative to one human (log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), fontsize=8.5, ncol=2)
    fig.subplots_adjust(top=0.96)
    vs.header(fig, "How much is one animal worth to an LLM? Implied exchange rates vs Rethink Priorities",
              "From forced choices between sparing N individuals of two species from a day of severe pain. "
              "95% bootstrap CI over comparisons.")
    vs.save(fig, FIG / "fig1_implied_weights.png")


def fig_vs_rp(allw, stats):
    vs.setup()
    models = sorted(allw.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.8 * len(models), 3.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        d = allw[(allw.model == m)].dropna(subset=["rp"])
        ax.scatter(d.rp, d.weight, s=40, color=vs.SERIES[0], edgecolor=vs.SURFACE, lw=1.2, zorder=3)
        lim = [1e-4, 3]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim[0] / 2, lim[1] * 2)
        ax.set_ylim(min(allw.weight.min(), lim[0]) / 3, lim[1] * 3)  # shared y: use the global minimum
        place_labels(ax, d.rp.values, d.weight.values, [s_.replace("bsf_larva", "BSF larva") for s_ in d.species])
        ax.plot(lim, lim, color=vs.AXIS, lw=1)
        ax.set_xlabel("Rethink Priorities welfare range")
        ax.set_title(f"{m}\nSpearman ρ = {stats[m]['spearman_vs_rp']:+.2f} (8 animal species)", fontsize=9.5)
    axes[0, 0].set_ylabel("LLM implied weight (human = 1)")
    vs.header(fig, "Do LLM exchange rates agree with Rethink Priorities' welfare ranges?",
              "Diagonal = agreement. Points below it: the model values that species less, relative to humans, than RP's median.")
    vs.save(fig, FIG / "fig2_vs_rethink_priorities.png")


def fig_slopes(allw):
    vs.setup()
    models = sorted(allw.model.unique())
    order = allw.groupby("species").log10_weight.mean().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.8))
    for k, m in enumerate(models):
        d = allw[allw.model == m].set_index("species").reindex(order)
        y = np.arange(len(order)) + (k - (len(models) - 1) / 2) * 0.2
        ax.scatter(d.slope, y, color=vs.SERIES[k], s=40, edgecolor=vs.SURFACE, lw=1.2, label=m, zorder=3)
    ax.axvline(0, color=vs.INK2, lw=1)
    ax.axvline(1, color=vs.AXIS, lw=1)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Relative scope slope (species slope / overall slope); 1 = average sensitivity to numbers")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), fontsize=8.5, ncol=3)
    fig.subplots_adjust(top=0.96)
    vs.header(fig, "Scope sensitivity by species: does sparing 10x more individuals matter?",
              "Per-species slope from the probit model. Near zero = numbers barely matter for that species.")
    vs.save(fig, FIG / "fig3_scope_slopes.png")


def main():
    r = load()
    allw, stats = [], {}
    n_prompts = sum(1 for _ in (ROOT / "data" / "pairs.jsonl").open(encoding="utf-8"))
    for m, d in r.groupby("model"):
        if len(d) < 0.95 * n_prompts:  # partial runs cover comparisons in order, which would bias the fit
            print(f"skip {m}: only {len(d)}/{n_prompts} prompts answered")
            continue
        w, st, boot = analyse_model(d)
        boot.to_csv(RES / f"boot_log10_weight__{m.split('/')[-1]}.csv", index=False, float_format="%.5g")
        w["model"] = m.split("/")[-1]
        allw.append(w)
        st["unparsed"] = int(d.choice.isna().sum())
        stats[m.split("/")[-1]] = st
    allw = pd.concat(allw)
    allw.to_csv(RES / "implied_weights.csv", index=False, float_format="%.5g")
    (RES / "summary.json").write_text(json.dumps(stats, indent=1, default=float))
    fig_weights(allw)
    fig_vs_rp(allw, stats)
    fig_slopes(allw)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(stats).T.round(3))
    print(allw.pivot(index="species", columns="model", values="weight").round(4))


if __name__ == "__main__":
    main()
