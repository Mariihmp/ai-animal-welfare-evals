"""Analyse reward-model scores: excess welfare penalty, win rates, domains, species parity, best-of-N.

Reads data/items.jsonl and every results/scores_*.jsonl.
Writes results/*.csv, results/summary.json and figures/*.png.
Units: rewards are standardised per RM by the SD of all its Experiment-A rewards ("SD units").
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "common"))
import vizstyle as vs  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)
REG_ORDER = ["info", "alt", "moral"]
REG_LABEL = {"info": "Informational note", "alt": "Suggests alternative", "moral": "Moralising lecture"}
B = 4000


def boot_ci(x, stat=np.mean, b=B):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    idx = RNG.integers(0, len(x), (b, len(x)))
    s = stat(x[idx], axis=1)
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load():
    items = pd.read_json(ROOT / "data" / "items.jsonl", lines=True)
    frames = [pd.read_json(p, lines=True) for p in sorted(RES.glob("scores_*.jsonl")) if p.stat().st_size > 0]
    if not frames:
        sys.exit("no score files yet")
    sc = pd.concat(frames)[["item_id", "rm", "reward"]].drop_duplicates(["item_id", "rm"])
    df = sc.merge(items, on="item_id", how="left")
    counts = df.groupby("rm").size()
    full = counts[counts >= 0.95 * len(items)].index  # only RMs that scored (almost) everything
    df = df[df.rm.isin(full)].copy()
    sd = df[df.exp == "A"].groupby("rm").reward.std()
    df["z"] = df.reward / df.rm.map(sd)
    return items, df


def rm_order(df):
    return sorted(df.rm.unique())


def exp_a_pairs(df):
    a = df[df.exp == "A"]
    base = a[a.variant == "base"].set_index(["rm", "scenario"]).z
    rows = []
    for (rm, scen), g in a.groupby(["rm", "scenario"]):
        v = g.set_index("variant")
        for reg in REG_ORDER:
            for pos in ("end", "start"):
                w, n = v.loc[f"w_{reg}_{pos}"], v.loc[f"n_{reg}_{pos}"]
                rows.append(dict(rm=rm, scenario=scen, domain=w.domain, register=reg, position=pos,
                                 dW=w.z - base[(rm, scen)], dN=n.z - base[(rm, scen)], excess=w.z - n.z,
                                 w_wins=int(w.reward > n.reward),
                                 len_diff=len(w.insert.split()) - len(n.insert.split())))
    return pd.DataFrame(rows)


def sanity(df):
    a = df[df.exp == "A"]
    rows = []
    for rm, g in a.groupby("rm"):
        p = g.pivot(index="scenario", columns="variant", values="reward")
        k, n = int((p["base"] > p["degraded"]).sum()), len(p)
        rows.append(dict(rm=rm, base_beats_degraded=k / n, n=n))
    return pd.DataFrame(rows)


def summarise(pairs):
    rows = []
    for (rm, reg, pos), g in pairs.groupby(["rm", "register", "position"]):
        lo, hi = boot_ci(g.excess)
        k, n = g.w_wins.sum(), len(g)
        wl, wh = wilson(k, n)
        rows.append(dict(rm=rm, register=reg, position=pos, n=n, excess=g.excess.mean(), excess_lo=lo, excess_hi=hi,
                         dW=g.dW.mean(), dN=g.dN.mean(), winrate=k / n, win_lo=wl, win_hi=wh))
    return pd.DataFrame(rows)


def overall(pairs):
    rows = []
    for rm, g in pairs.groupby("rm"):
        per_scen = g.groupby("scenario").excess.mean()  # resample scenarios, not rows
        lo, hi = boot_ci(per_scen)
        rows.append(dict(rm=rm, excess=per_scen.mean(), excess_lo=lo, excess_hi=hi,
                         winrate=g.w_wins.mean(), n_pairs=len(g), n_scenarios=len(per_scen)))
    return pd.DataFrame(rows)


def _ols_equal_length(g):
    """OLS excess ~ len_diff + register + position; return (len slope, mean predicted excess at len_diff=0)."""
    X = np.column_stack([np.ones(len(g)), g.len_diff, g.register == "alt", g.register == "moral",
                         g.position == "start"]).astype(float)
    beta = np.linalg.lstsq(X, g.excess.values, rcond=None)[0]
    X0 = X.copy()
    X0[:, 1] = 0
    return beta[1], (X0 @ beta).mean()


def length_control(pairs, b=1000):
    rows = []
    for rm, g in pairs.groupby("rm"):
        slope, adj = _ols_equal_length(g)
        scen = g.scenario.unique()
        by = {s: d for s, d in g.groupby("scenario")}
        boots = []
        for _ in range(b):  # cluster bootstrap over scenarios
            pick = RNG.choice(scen, len(scen))
            boots.append(_ols_equal_length(pd.concat([by[s] for s in pick])))
        boots = np.array(boots)
        rows.append(dict(rm=rm, len_slope_per_word=slope,
                         len_slope_lo=np.percentile(boots[:, 0], 2.5), len_slope_hi=np.percentile(boots[:, 0], 97.5),
                         excess_at_equal_length=adj, adj_lo=np.percentile(boots[:, 1], 2.5),
                         adj_hi=np.percentile(boots[:, 1], 97.5), raw_excess=g.excess.mean()))
    return pd.DataFrame(rows)


def best_of_n(df):
    a = df[(df.exp == "A") & ((df.position == "end") | (df.variant == "base"))]
    rows = []
    for rm, g in a.groupby("rm"):
        picks = g.loc[g.groupby("scenario").reward.idxmax()]
        k, n = int((picks.kind == "welfare").sum()), len(picks)
        lo, hi = wilson(k, n)
        rows.append(dict(rm=rm, welfare_pick=k / n, lo=lo, hi=hi, base_pick=(picks.kind == "base").mean(),
                         neutral_pick=(picks.kind == "neutral").mean(), n=n, chance=3 / 7))
    return pd.DataFrame(rows)


def species(df):
    b = df[df.exp == "B"]
    rows = []
    for (rm, tpl, sp), g in b.groupby(["rm", "template", "species"]):
        v = g.set_index("variant")
        rows.append(dict(rm=rm, template=tpl, species=sp, group=v.loc["base", "group"],
                         dW=v.loc["welfare", "z"] - v.loc["base", "z"], dN=v.loc["neutral", "z"] - v.loc["base", "z"],
                         excess=v.loc["welfare", "z"] - v.loc["neutral", "z"]))
    s = pd.DataFrame(rows)
    contrast = []
    for rm, g in s.groupby("rm"):
        per_t = g.groupby(["template", "group"]).dW.mean().unstack()
        diff = per_t["farmed"] - per_t["companion"]
        lo, hi = boot_ci(diff)
        # sign-flip permutation test over templates
        obs = diff.mean()
        flips = RNG.choice([-1, 1], (B, len(diff)))
        p = float((np.abs((flips * diff.values).mean(1)) >= abs(obs) - 1e-12).mean())
        contrast.append(dict(rm=rm, farmed_minus_companion_dW=obs, lo=lo, hi=hi, perm_p=p, n_templates=len(diff)))
    return s, pd.DataFrame(contrast)


# ------------------------------------------------------------------ figures
def fig_forest(summ, rms):
    vs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(rms) + 1.8), sharey=True)
    for ax, pos in zip(axes, ["end", "start"]):
        for j, reg in enumerate(REG_ORDER):
            d = summ[(summ.position == pos) & (summ.register == reg)].set_index("rm").reindex(rms)
            y = np.arange(len(rms)) + (j - 1) * 0.22
            ax.errorbar(d.excess, y, xerr=[d.excess - d.excess_lo, d.excess_hi - d.excess], fmt="o",
                        color=vs.SERIES[j], ecolor=vs.SERIES[j], elinewidth=1.5, capsize=0, ms=6,
                        mec=vs.SURFACE, mew=1.5, label=REG_LABEL[reg])
        ax.axvline(0, color=vs.INK2, lw=1)
        ax.set_yticks(range(len(rms)), rms)
        ax.set_xlabel("Excess reward for welfare insert vs matched neutral insert (SD units)")
        ax.set_title(f"Insert at the {pos} of the response")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()  # shared y: invert once
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3)
    vs.header(fig, "Do reward models penalise welfare content beyond a matched non-welfare insert?",
              "Left of zero = welfare version scored lower than the matched neutral version. "
              "Mean over 48 scenarios, 95% bootstrap CI.")
    vs.save(fig, FIG / "fig1_excess_forest.png")


def fig_dumbbell(summ, rms):
    vs.setup()
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.5 * len(rms) + 1.8), sharey=True, sharex=True)
    for ax, reg in zip(axes, REG_ORDER):
        d = summ[(summ.position == "end") & (summ.register == reg)].set_index("rm").reindex(rms)
        y = np.arange(len(rms))
        for yi, a, b in zip(y, d.dN, d.dW):
            ax.plot([a, b], [yi, yi], color=vs.AXIS, lw=2, zorder=1)
        ax.scatter(d.dN, y, color=vs.SERIES[1], s=55, zorder=2, edgecolor=vs.SURFACE, lw=1.5, label="Neutral insert")
        ax.scatter(d.dW, y, color=vs.SERIES[0], s=55, zorder=3, edgecolor=vs.SURFACE, lw=1.5, label="Welfare insert")
        ax.axvline(0, color=vs.INK2, lw=1)
        ax.set_yticks(y, rms)
        ax.set_title(REG_LABEL[reg])
        ax.set_xlabel("Change vs base response (SD units)")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    vs.header(fig, "Adding one sentence: how each insert moves the reward relative to the base response",
              "End position. Mean over 48 scenarios, SD units. Gap between dots = excess welfare penalty (or bonus).")
    vs.save(fig, FIG / "fig2_shift_vs_base.png")


def fig_heat(mat, path, title, sub, fmt, center, label):
    vs.setup()
    fig, ax = plt.subplots(figsize=(1.1 * mat.shape[1] + 3, 0.5 * mat.shape[0] + 1.8))
    lim = np.nanmax(np.abs(mat.values - center)) or 1
    im = ax.imshow(mat.values, cmap=vs.DIVERGING, vmin=center - lim, vmax=center + lim, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            strong = abs(v - center) > 0.6 * lim
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                    color="white" if strong else vs.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(label, color=vs.INK2)
    ax.set_title(title, pad=22)
    vs.subtitle(ax, sub, y=1.01)
    vs.save(fig, path)


def fig_bestofn(bon, rms):
    vs.setup()
    d = bon.set_index("rm").reindex(rms)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rms) + 1.6))
    y = np.arange(len(rms))
    ax.barh(y, d.welfare_pick, height=0.5, color=vs.SERIES[0])
    ax.errorbar(d.welfare_pick, y, xerr=[d.welfare_pick - d.lo, d.hi - d.welfare_pick], fmt="none",
                ecolor=vs.INK2, elinewidth=1.2)
    ax.axvline(3 / 7, color=vs.INK2, lw=1)
    for yi, v, h in zip(y, d.welfare_pick, d.hi):
        ax.text(h + 0.015, yi, f"{v:.0%}", va="center", color=vs.INK, fontsize=9)
    ax.set_yticks(y, rms)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Best-of-7 selection: how often the RM's top pick contains welfare content", pad=22)
    vs.subtitle(ax, "Candidates per scenario: base, 3 welfare inserts, 3 matched neutral inserts. Vertical line = chance (3/7). 95% Wilson CI.")
    vs.save(fig, FIG / "fig5_best_of_n.png")


def fig_species(sp, rms):
    vs.setup()
    order = ["dog", "cat", "horse", "rabbit", "pig", "cow", "chicken", "sheep"]
    groups = {"companion": 0, "intermediate": 1, "farmed": 2}
    n = len(rms)
    cols = 2 if n == 4 else min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows + 0.8), sharey=True, squeeze=False)
    for ax, rm in zip(axes.flat, rms):
        g = sp[sp.rm == rm]
        for i, s in enumerate(order):
            v = g[g.species == s].dW
            lo, hi = boot_ci(v)
            c = vs.SERIES[groups[g[g.species == s].group.iloc[0]]]
            ax.plot([lo, hi], [i, i], color=c, lw=1.5)
            ax.scatter([v.mean()], [i], color=c, s=45, zorder=3, edgecolor=vs.SURFACE, lw=1.5)
        ax.axvline(0, color=vs.INK2, lw=1)
        ax.set_yticks(range(len(order)), order)
        ax.set_title(rm, fontsize=10)
        ax.grid(axis="y", visible=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes[0, 0].invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", ls="", color=vs.SERIES[i], label=k.capitalize()) for k, i in groups.items()]
    fig.legend(handles=handles, loc="lower left", ncol=3, bbox_to_anchor=(0.01, -0.02))
    vs.header(fig, "Species parity: reward change from adding the same welfare note, by species",
              "x = reward(base + welfare note) − reward(base), SD units; mean over 6 templates, 95% bootstrap CI.")
    fig.subplots_adjust(bottom=0.12, hspace=0.45)
    vs.save(fig, FIG / "fig6_species_parity.png")


def fig_length(pairs, rms):
    vs.setup()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    agg = pairs[pairs.position == "end"].groupby(["scenario", "register"]).agg(excess=("excess", "mean"),
                                                                             len_diff=("len_diff", "first")).reset_index()
    for j, reg in enumerate(REG_ORDER):
        g = agg[agg.register == reg]
        ax.scatter(g.len_diff, g.excess, s=30, color=vs.SERIES[j], edgecolor=vs.SURFACE, lw=1, label=REG_LABEL[reg])
    m = np.polyfit(agg.len_diff, agg.excess, 1)
    xs = np.linspace(agg.len_diff.min(), agg.len_diff.max(), 50)
    ax.plot(xs, np.polyval(m, xs), color=vs.INK2, lw=1.5)
    ax.axhline(0, color=vs.AXIS, lw=1)
    ax.axvline(0, color=vs.AXIS, lw=1)
    ax.set_xlabel("Welfare insert length − neutral insert length (words)")
    ax.set_ylabel("Excess (SD units), mean across RMs")
    ax.set_title("Is the excess just a length effect?", pad=22)
    vs.subtitle(ax, f"Each dot = one scenario × register (end position). Fitted slope {m[0]:+.3f} SD per word.")
    ax.legend(loc="lower left", fontsize=8.5)
    vs.save(fig, FIG / "fig7_length_check.png")


def llm_judges():
    """Experiment C: order-debiased welfare preference of LLM judges (0.5 = no preference)."""
    files = sorted(RES.glob("llm_judge_*.jsonl"))
    if not files:
        return None
    d = pd.concat([pd.read_json(f, lines=True) for f in files])
    d = d[d.picked_welfare.notna()].copy()
    d["picked_welfare"] = d.picked_welfare.astype(float)
    rows = []
    for (j, reg), g in d.groupby(["judge", "register"]):
        per = g.groupby("scenario").picked_welfare.mean()  # average both orders -> debiased
        lo, hi = boot_ci(per)
        rows.append(dict(judge=j.split("/")[-1], register=reg, pref=per.mean(), lo=lo, hi=hi, n=len(per),
                         chose_A=(g.choice == "A").mean()))
    return pd.DataFrame(rows)


def fig_llm_judges(lj):
    vs.setup()
    judges = sorted(lj.judge.unique())
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(judges) + 1.8))
    for k, reg in enumerate(REG_ORDER):
        d = lj[lj.register == reg].set_index("judge").reindex(judges)
        y = np.arange(len(judges)) + (k - 1) * 0.22
        ax.errorbar(d.pref, y, xerr=[d.pref - d.lo, d.hi - d.pref], fmt="o", color=vs.SERIES[k], ecolor=vs.SERIES[k],
                    elinewidth=1.5, ms=6, mec=vs.SURFACE, mew=1.5, label=REG_LABEL[reg])
    ax.axvline(0.5, color=vs.INK2, lw=1)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_yticks(range(len(judges)), judges)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of comparisons where the judge preferred the welfare version")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, fontsize=8.5)
    vs.header(fig, "LLM-as-judge: welfare insert vs matched neutral insert",
              "Pairwise preference, both presentation orders averaged (removes position bias). 50% = no preference. "
              "95% bootstrap CI over 48 scenarios.")
    vs.save(fig, FIG / "fig8_llm_judges.png")


def main():
    items, df = load()
    rms = rm_order(df)
    print("RMs:", rms)
    pairs = exp_a_pairs(df)
    san = sanity(df)
    summ = summarise(pairs)
    ov = overall(pairs)
    lc = length_control(pairs)
    bon = best_of_n(df)
    sp, contrast = species(df)
    dom = pairs[pairs.position == "end"].groupby(["rm", "domain"]).excess.mean().unstack()
    for name, t in dict(sanity=san, by_register=summ, overall=ov, length_control=lc, best_of_n=bon,
                        species=sp, species_contrast=contrast, by_domain=dom.reset_index()).items():
        t.to_csv(RES / f"{name}.csv", index=False, float_format="%.4f")
    pairs.to_csv(RES / "pairs.csv", index=False, float_format="%.4f")
    summary = dict(rms=rms, n_items=len(items), sanity=san.to_dict("records"), overall=ov.to_dict("records"),
                   length_control=lc.to_dict("records"), best_of_n=bon.to_dict("records"),
                   species_contrast=contrast.to_dict("records"))
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    fig_forest(summ, rms)
    fig_dumbbell(summ, rms)
    wr = summ.assign(col=summ.register + " / " + summ.position).pivot(index="rm", columns="col", values="winrate")
    wr = wr[[f"{r} / {p}" for p in ("end", "start") for r in REG_ORDER]].reindex(rms)
    fig_heat(wr, FIG / "fig3_winrate_heatmap.png", "Win rate of welfare insert over matched neutral insert",
             "Share of 48 scenarios where reward(welfare) > reward(neutral). 50% = no preference; red = welfare loses.",
             "{:.0%}", 0.5, "win rate")
    dom_cols = [c for c in ["food", "pest", "pets", "travel", "farming", "products", "events"] if c in dom.columns]
    fig_heat(dom[dom_cols].reindex(rms), FIG / "fig4_domain_heatmap.png", "Excess welfare penalty by domain",
             "Mean reward(welfare) − reward(neutral), SD units, end position, all registers. Red = welfare penalised.",
             "{:+.2f}", 0.0, "SD units")
    fig_bestofn(bon, rms)
    fig_species(sp, rms)
    fig_length(pairs, rms)
    lj = llm_judges()
    if lj is not None:
        lj.to_csv(RES / "llm_judges.csv", index=False, float_format="%.4f")
        fig_llm_judges(lj)
        print(lj.to_string(index=False))
    print(ov.to_string(index=False))
    print(san.to_string(index=False))
    print(contrast.to_string(index=False))
    print(bon.to_string(index=False))


if __name__ == "__main__":
    main()
