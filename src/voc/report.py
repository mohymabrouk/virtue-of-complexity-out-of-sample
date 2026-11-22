"""Report stage: tables (results/tables.md), figures and README.md, built only from the files in results/."""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from voc import config, metrics, pipeline  # noqa: E402

log = logging.getLogger(__name__)
R = config.RESULTS

# reference palette (dataviz skill): ordinal blue ramp for z, categorical slots for strategies
Z_COLORS = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CONV_LABEL = {"std": "standardised returns (KMZ spec, primary)", "raw": "raw returns (Nagel's convention, secondary)"}
TARGET = pipeline.voc_name(config.TARGET_P, config.TARGET_Z)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _f(x, d=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _r2(x):
    if x is None or not np.isfinite(x):
        return "–"
    return "< -100%" if x < -1 else _pct(x)


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# Part A tables
# ---------------------------------------------------------------------------------------------------------------
def load_partA():
    paper = pd.read_csv(R / "partA_paper_table.csv")
    post = pd.read_csv(R / "partA_post_table.csv") if (R / "partA_post_table.csv").exists() else None
    return paper, post


def main_table(conv: str) -> str:
    paper, post = load_partA()
    p = paper[(paper.conv == conv)].set_index("strategy")
    q = post[(post.conv == conv) & (post.period == "post")].set_index("strategy") if post is not None else None
    rows = []
    for name in pipeline.main_strategy_names():
        row = [name]
        for t in (p, q):
            if t is None or name not in t.index:
                row += ["–"] * 4
                continue
            r = t.loc[name]
            row += [f"{r.sharpe:.2f} [{r.sharpe_lo:.2f}, {r.sharpe_hi:.2f}]", _f(r.dsr), _f(r.alpha_t, 1), _r2(r.r2)]
        rows.append(row)
    n1 = int(p["n"].iloc[0])
    n2 = int(q["n"].iloc[0]) if q is not None else 0
    header = ["Strategy", f"Sharpe [95% CI], 1931-2020 (n={n1})", "DSR", "α t", "OOS R²",
              f"Sharpe [95% CI], 2021-01..2026-01 (n={n2})", "DSR", "α t", "OOS R²"]
    return md_table(header, rows)


def dsharpe_table(conv: str) -> str:
    paper, post = load_partA()
    rows = []
    for name in pipeline.main_strategy_names()[1:]:
        row = [name]
        for t, per in ((paper, "paper"), (post, "post")):
            if t is None:
                row.append("–")
                continue
            r = t[(t.conv == conv) & (t.period == per) & (t.strategy == name)].iloc[0]
            row.append(f"{r.dsharpe:+.2f} [{r.dsharpe_lo:+.2f}, {r.dsharpe_hi:+.2f}]")
            row.append(f"{r.ir:.2f}")
        rows.append(row)
    return md_table(["Strategy", "ΔSharpe vs buy-and-hold, 1931-2020 [block-bootstrap 95% CI]", "IR",
                     "ΔSharpe, 2021-01..2026-01 [CI]", "IR"], rows)


def replication_table() -> str:
    chk = json.loads((R / "replication_check.json").read_text())
    paper, _ = load_partA()
    s = paper[paper.conv == "std"].set_index("strategy")
    rw = paper[paper.conv == "raw"].set_index("strategy")
    t1 = config.PAPER_TABLE1
    cur = pd.read_csv(R / "partA_paper_curve.csv")
    cs = cur[(cur.conv == "std") & (cur.P == 12000) & (cur.z == 1e-3)].iloc[0]
    rows = [
        ["VoC P=12000, z=10³: Sharpe (mean over seeds)", f"{t1['nonlinear_z1e3']['SR']:.2f}", f"{chk['ours_sharpe_mean_over_seeds']:.2f}",
         f"{chk['raw_convention']['sharpe']:.2f}"],
        ["VoC P=12000, z=10³: α t-stat (mean over seeds)", f"{t1['nonlinear_z1e3']['IR_t']:.1f}", f"{chk['ours_alpha_t_mean_over_seeds']:.1f}",
         f"{chk['raw_convention']['alpha_t']:.1f}"],
        ["VoC P=12000, z=10³: information ratio", f"{t1['nonlinear_z1e3']['IR']:.2f}", f"{chk['ours_ir_mean_over_seeds']:.2f}",
         f"{chk['raw_convention']['ir']:.2f}"],
        ["VoC P=12000, z=10³: OOS R² (mean over seeds)", _pct(t1["nonlinear_z1e3"]["R2"]), _pct(chk["ours_r2_mean_over_seeds"]), "–"],
        ["VoC P=12000, ridgeless (z=10⁻³): α t / IR (mean over seeds)", "Nagel (raw, KMZ code): 2.42 / 0.26; KMZ Fig. 8 (std): 2.81 / 0.30",
         f"{cs.alpha_t_paper:.2f} / {cs.ir_paper:.2f}",
         f"{chk['raw_convention_ridgeless']['alpha_t']:.2f} / {chk['raw_convention_ridgeless']['ir']:.2f}"],
        ["Linear ridge z=10³: Sharpe / α t (no seeds involved)", f"{t1['linear_z1e3']['SR']:.2f} / {t1['linear_z1e3']['IR_t']:.1f}",
         f"{s.loc['Linear ridge, z=1e3', 'sharpe']:.2f} / {s.loc['Linear ridge, z=1e3', 'alpha_t']:.1f}",
         f"{rw.loc['Linear ridge, z=1e3', 'sharpe']:.2f} / {rw.loc['Linear ridge, z=1e3', 'alpha_t']:.1f}"],
        ["Linear ridgeless (kitchen sink): Sharpe", f"{t1['linear_ridgeless']['SR']:.2f}", f"{s.loc['Linear ridge, z=0+', 'sharpe']:.2f}",
         f"{rw.loc['Linear ridge, z=0+', 'sharpe']:.2f}"],
        ["Nagel vol-timed momentum: α t / IR (raw)", "Nagel Table I: 3.68 / 0.39", "–",
         f"{rw.loc['Nagel vol-timed momentum', 'alpha_t']:.2f} / {rw.loc['Nagel vol-timed momentum', 'ir']:.2f}"],
    ]
    return md_table(["Quantity (1931-2020, T=12)", "Published", "Ours, standardised (primary)", "Ours, raw returns"], rows)


def curve_table(period_file: str, period: str, conv: str) -> str:
    c = pd.read_csv(R / period_file)
    if "period" in c.columns:
        c = c[c.period == period]
    c = c[c.conv == conv]
    rows = []
    for P in config.P_GRID:
        r = c[(c.P == P) & (c.z == config.MAIN_Z)].iloc[0]
        r0 = c[(c.P == P) & (c.z == 1e-3)].iloc[0]
        rows.append([P, f"{P / config.T:g}", _f(r0.sharpe_paper), _f(r.sharpe_paper), _f(r.sharpe_ens), _f(r.alpha_t_paper, 1),
                     _f(r.alpha_t_ens, 1), _r2(r0.r2_paper), _r2(r.r2_paper)])
    return md_table(["P", "c", "Sharpe z=10⁻³ (paper conv.)", "Sharpe z=10³ (paper conv.)", "Sharpe z=10³ (ensemble)",
                     "α t z=10³ (paper)", "α t z=10³ (ensemble)", "R² z=10⁻³", "R² z=10³"], rows)


# ---------------------------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------------------------
def fig_curves(curve: pd.DataFrame, conv: str, title: str, path, bh_sharpe: float | None = None) -> None:
    c = curve[curve.conv == conv]
    panels = [("sharpe_paper", "Sharpe, mean over seeds (paper)"), ("sharpe_ens", "Sharpe, seed-averaged forecast (ensemble)"),
              ("alpha_t_paper", "α t-stat vs market, mean over seeds"), ("r2_paper", "OOS R², mean over seeds")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    for ax, (col, lab) in zip(axes.ravel(), panels):
        _style(ax)
        for j, z in enumerate(config.Z_GRID):
            d = c[c.z == z].sort_values("c")
            y = d[col].to_numpy()
            if col == "r2_paper":
                y = np.clip(y, -0.10, None)
            ax.plot(d["c"], y, color=Z_COLORS[j], linewidth=2, label=f"z = 1e{int(np.log10(z))}")
        ax.set_xscale("log")
        ax.axvline(1, color=INK2, linewidth=0.8, linestyle=":")
        if col.startswith("sharpe") and bh_sharpe is not None:
            ax.axhline(bh_sharpe, color=CAT[1], linewidth=1.5, linestyle="--", label="buy-and-hold")
        if col == "alpha_t_paper":
            ax.axhline(1.96, color=INK2, linewidth=0.8, linestyle="--")
        if col == "r2_paper":
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.set_ylim(-0.105, None)
            lab += " (clipped at -10%)"
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("complexity c = P / T (log scale; dotted line: c = 1)", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_post_cumulative(path) -> None:
    pos = pd.read_parquet(R / "partA_post_positions.parquet")
    names = ["Buy-and-hold", TARGET, "Linear ridge, z=1e3", "Nagel vol-timed momentum"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, conv in zip(axes, config.CONVENTIONS):
        _style(ax)
        d = pos[pos.conv == conv]
        months = pd.PeriodIndex(d["month"], freq="M").to_timestamp()
        bh_vol = d["y"].std()
        for k, name in enumerate(names):
            ret = d[name].to_numpy() * d["y"].to_numpy()
            ret = ret / ret.std() * bh_vol  # same ex-post volatility as buy-and-hold, display only
            ax.plot(months, np.cumsum(ret), color=CAT[k], linewidth=2, label=name)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"Post-sample, {CONV_LABEL[conv]}", fontsize=9, color=INK, loc="left")
        ax.set_ylabel("cumulative return, scaled to buy-and-hold vol", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_positions(path) -> None:
    pos = pd.read_parquet(R / "partA_paper_positions.parquet")
    post = pd.read_parquet(R / "partA_post_positions.parquet")
    d = pd.concat([pos, post])
    d = d[d.conv == "std"].drop_duplicates("month").set_index("month").sort_index()
    months = pd.PeriodIndex(d.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURFACE)
    _style(ax)
    for k, name in enumerate([TARGET, "Nagel vol-timed momentum"]):
        x = d[name] / d[name].std()
        ax.plot(months, x.rolling(6).mean(), color=CAT[k], linewidth=1.5, label=f"{name} (6-month mean, unit sd)")
    corr = d[[TARGET, "Nagel vol-timed momentum"]].corr().iloc[0, 1]
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(pd.Timestamp("2021-01-01"), color=INK2, linewidth=0.8, linestyle=":")
    ax.set_title(f"US timing positions, standardised spec (monthly correlation {corr:.2f}); dotted line: end of the paper's sample",
                 fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------------------------------------------
def partB_frames():
    t = pd.read_csv(R / "partB_country_table.csv")
    bh = t[t.strategy == "Buy-and-hold"][["location", "conv", "sharpe"]].rename(columns={"sharpe": "bh"})
    t = t.merge(bh, on=["location", "conv"])
    t["d_sr"] = t["sharpe"] - t["bh"]
    return t


def partB_table(conv: str) -> str:
    t = partB_frames()
    t = t[(t.conv == conv) & (t.location != "usa")]
    n = t.location.nunique()
    rows = []
    for name in pipeline.main_strategy_names():
        d = t[t.strategy == name]
        q1, med, q3 = np.percentile(d.sharpe, [25, 50, 75])
        rows.append([name, f"{med:.2f} [{q1:.2f}, {q3:.2f}]", f"{d.d_sr.median():+.2f}",
                     "–" if name == "Buy-and-hold" else f"{(d.d_sr > 0).sum()}/{n}",
                     "–" if name == "Buy-and-hold" else f"{d.alpha_t.median():.2f}",
                     "–" if name == "Buy-and-hold" else f"{(d.alpha_t > 1.96).sum()} / {(d.alpha_t < -1.96).sum()}",
                     _r2(d.r2.median()) if d.r2.notna().any() else "–"])
    return md_table(["Strategy", "Sharpe: median [IQR]", "median ΔSharpe vs B&H", "beats B&H", "median α t",
                     "α t > 1.96 / < -1.96", "median OOS R²"], rows), n


def partB_counts(conv: str) -> str:
    t = partB_frames()
    t = t[(t.conv == conv) & (t.location != "usa")].set_index(["location", "strategy"])
    locs = sorted(set(t.index.get_level_values(0)))
    rows = []
    for other in ["Buy-and-hold", "Linear ridge, z=1e3", "Linear ridge, z=0+", "TSMOM sign", "Mom12 (equal weights)",
                  "Nagel vol-timed momentum", "KMZ momentum (mean/var)", pipeline.voc_name(12, 1000.0)]:
        wins = sum(t.loc[(loc, TARGET), "sharpe"] > t.loc[(loc, other), "sharpe"] for loc in locs)
        p = binomtest(int(wins), len(locs), 0.5).pvalue
        rows.append([other, f"{wins}/{len(locs)}", f"{p:.3f}"])
    return md_table([f"{TARGET} has a higher Sharpe than ...", "countries", "sign-test p (two-sided)"], rows)


def partB_pooled_table(conv: str) -> str:
    p = pd.read_csv(R / "partB_pooled_table.csv")
    p = p[p.conv == conv].set_index("strategy")
    rows = []
    for name in pipeline.main_strategy_names():
        r = p.loc[name]
        rows.append([name, f"{r.sharpe:.2f} [{r.sharpe_lo:.2f}, {r.sharpe_hi:.2f}]", _f(r.dsr), _f(r.alpha_t, 1),
                     "–" if name == "Buy-and-hold" else f"{r.dsharpe:+.2f} [{r.dsharpe_lo:+.2f}, {r.dsharpe_hi:+.2f}]"])
    first, last = p["first"].iloc[0], p["last"].iloc[0]
    return md_table(["Strategy (equal-weighted across non-US countries)", "Sharpe [95% CI]", "DSR (13 trials)",
                     "α t vs EW B&H", "ΔSharpe vs EW B&H [CI]"], rows), first, last


def us_bridge_table() -> tuple[str, int]:
    """US with JKP predictors vs US with Welch-Goyal predictors over the same months (Part A positions)."""
    t = partB_frames()
    t = t[t.location == "usa"].set_index(["conv", "strategy"])
    rets = pd.read_parquet(R / "partB_strategy_returns.parquet")
    months = set(rets[rets.location == "usa"]["month"])
    a = pd.concat([pd.read_parquet(R / "partA_paper_positions.parquet"), pd.read_parquet(R / "partA_post_positions.parquet")])
    a = a[a["month"].isin(months)]
    names = ["Buy-and-hold", "Linear ridge, z=1e3", "Nagel vol-timed momentum", TARGET]
    rows = []
    for name in names:
        row = [name]
        for c in config.CONVENTIONS:
            row.append(f"{t.loc[(c, name), 'sharpe']:.2f} (α t {_f(t.loc[(c, name), 'alpha_t'], 1)})")
        for c in config.CONVENTIONS:
            g = a[a.conv == c]
            ret = g[name].to_numpy() * g["y"].to_numpy()
            sr = metrics.sharpe(pd.Series(ret), 12)
            at = "–" if name == "Buy-and-hold" else _f(metrics.alpha_regression(ret, g["y"].to_numpy())["alpha_t"], 1)
            row.append(f"{sr:.2f} (α t {at})")
        rows.append(row)
    first, last = min(months), max(months)
    hdr = ["Strategy", "JKP predictors, std", "JKP predictors, raw", "Welch-Goyal predictors, std", "Welch-Goyal predictors, raw"]
    note = f"Same months for all four columns: {first} to {last} ({len(months)} months)."
    return md_table(hdr, rows) + "\n\n" + note, len(months)


def fig_partB(path) -> None:
    cur = pd.read_csv(R / "partB_curves.csv")
    t = partB_frames()
    bh = t[t.strategy == "Buy-and-hold"].set_index(["location", "conv"])["sharpe"]
    cur = cur[cur.location != "usa"].copy()
    cur["bh"] = [bh.loc[(loc, c)] for loc, c in zip(cur.location, cur.conv)]
    cur["d_ens"] = cur["sharpe_ens"] - cur["bh"]
    cur["d_paper"] = cur["sharpe_paper"] - cur["bh"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    for i, conv in enumerate(config.CONVENTIONS):
        for j, (col, lab) in enumerate([("d_paper", "mean over seeds"), ("d_ens", "ensemble")]):
            ax = axes[i, j]
            _style(ax)
            for k, z in enumerate(config.Z_GRID):
                d = cur[(cur.conv == conv) & (cur.z == z)].groupby("c")[col].median()
                ax.plot(d.index, d.values, color=Z_COLORS[k], linewidth=2, label=f"z = 1e{int(np.log10(z))}")
            ax.axhline(0, color=CAT[1], linewidth=1.5, linestyle="--")
            ax.set_xscale("log")
            ax.axvline(1, color=INK2, linewidth=0.8, linestyle=":")
            ax.set_title(f"Median ΔSharpe vs buy-and-hold, {lab}; {conv}", fontsize=9, color=INK, loc="left")
            ax.set_xlabel("complexity c = P / T", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Part B: 41 non-US JKP markets, same specification everywhere", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_partB_countries(path) -> None:
    t = partB_frames()
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), facecolor=SURFACE, sharey=True)
    base = t[(t.conv == "std") & (t.strategy == TARGET)].sort_values("d_sr")
    order = list(base.location)
    for ax, conv in zip(axes, config.CONVENTIONS):
        _style(ax)
        d = t[(t.conv == conv) & (t.strategy == TARGET)].set_index("location").loc[order]
        v = t[(t.conv == conv) & (t.strategy == "Nagel vol-timed momentum")].set_index("location").loc[order]
        yy = np.arange(len(order))
        ax.barh(yy, d.d_sr, color=CAT[0], height=0.7, label=f"{TARGET} minus B&H")
        ax.scatter(v.d_sr, yy, color=CAT[1], s=14, zorder=3, label="Nagel vol-timed momentum minus B&H")
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(yy)
        ax.set_yticklabels([("USA (bridge)" if o == "usa" else o.upper()) for o in order], fontsize=6)
        ax.set_title(f"ΔSharpe vs buy-and-hold, {conv}", fontsize=9, color=INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def run() -> None:
    parts = ["# Results tables\n", "Built by `python -m voc.cli report` from the CSV files in `results/`.\n"]
    for conv in config.CONVENTIONS:
        parts += [f"\n## Main performance table, US, {CONV_LABEL[conv]}\n", main_table(conv) + "\n",
                  f"\n### ΔSharpe vs buy-and-hold, {conv}\n", dsharpe_table(conv) + "\n"]
    parts += ["\n## Replication check\n", replication_table() + "\n"]
    for conv in config.CONVENTIONS:
        parts += [f"\n## Complexity curve at z=10³ and z=10⁻³, 1931-2020, {conv}\n", curve_table("partA_paper_curve.csv", "paper", conv) + "\n"]
        if (R / "partA_post_curve.csv").exists():
            parts += [f"\n## Complexity curve, post-sample 2021-01..2026-01, {conv}\n", curve_table("partA_post_curve.csv", "post", conv) + "\n"]
    paper_curve = pd.read_csv(R / "partA_paper_curve.csv")
    paper_tab, post_tab = load_partA()
    for conv in config.CONVENTIONS:
        bh = paper_tab[(paper_tab.conv == conv) & (paper_tab.strategy == "Buy-and-hold")].sharpe.iloc[0]
        fig_curves(paper_curve, conv, f"US, 1931-2020, T=12, {CONV_LABEL[conv]}", R / f"fig_curve_paper_{conv}.png", bh)
    if post_tab is not None:
        post_curve = pd.read_csv(R / "partA_post_curve.csv")
        post_curve = post_curve[post_curve.period == "post"]
        for conv in config.CONVENTIONS:
            bh = post_tab[(post_tab.conv == conv) & (post_tab.period == "post") & (post_tab.strategy == "Buy-and-hold")].sharpe.iloc[0]
            fig_curves(post_curve, conv, f"US, post-sample 2021-01..2026-01, {CONV_LABEL[conv]}", R / f"fig_curve_post_{conv}.png", bh)
        fig_post_cumulative(R / "fig_post_cumulative.png")
        fig_positions(R / "fig_positions_us.png")
    if (R / "partB_country_table.csv").exists():
        for conv in config.CONVENTIONS:
            tab, n = partB_table(conv)
            pooled, first, last = partB_pooled_table(conv)
            parts += [f"\n## Part B, {CONV_LABEL[conv]}: distribution across {n} non-US countries\n", tab + "\n",
                      f"\n### Head-to-head counts, {conv}\n", partB_counts(conv) + "\n",
                      f"\n### Pooled: equal-weighted portfolio across countries, {first}..{last}, {conv}\n", pooled + "\n"]
        bridge, _ = us_bridge_table()
        parts += ["\n## US with JKP predictors (bridge between Part A and Part B)\n", bridge + "\n"]
        fig_partB(R / "fig_partB_curves.png")
        fig_partB_countries(R / "fig_partB_countries.png")
    parts.insert(2, "\n## Post-sample behaviour of the target VoC cell, by year (standardised)\n")
    parts.insert(3, post_by_year("std")[0] + "\n") if post_tab is not None else None
    (R / "tables.md").write_text(no_negative_zero("\n".join(parts)), encoding="utf-8")
    log.info("wrote %s", R / "tables.md")
    try:
        from voc import readme
        readme.write()
    except ImportError:
        log.info("readme module not written yet")


def post_by_year(conv: str = "std") -> tuple[str, dict]:
    """Post-sample behaviour of the target VoC cell, year by year (positions relative to the 1931-2020 mean)."""
    d = pd.read_parquet(R / "partA_post_positions.parquet")
    p = pd.read_parquet(R / "partA_paper_positions.parquet")
    d, p = d[d.conv == conv].copy(), p[p.conv == conv]
    d["year"] = d["month"].str[:4]
    d["voc"] = d[TARGET] * d["y"]
    d["vm"] = d["Nagel vol-timed momentum"] * d["y"]
    base = p[TARGET].mean()
    rows, facts = [], {}
    for yr, g in d.groupby("year"):
        rows.append([yr, len(g), f"{g[TARGET].mean() / base:+.2f}", f"{(g[TARGET] > 0).sum()}/{len(g)}",
                     f"{g['y'].sum():+.2f}", f"{g['voc'].sum() / p[TARGET].abs().mean():+.2f}"])
        facts[yr] = {"rel_pos": g[TARGET].mean() / base, "n_long": int((g[TARGET] > 0).sum()), "n": len(g),
                     "bh": g["y"].sum(), "voc": g["voc"].sum() / p[TARGET].abs().mean()}
    worst = d.loc[d["voc"].idxmin()]
    facts["worst_month"] = {"month": worst["month"], "rel_pos": worst[TARGET] / base, "y": worst["y"]}
    tab = md_table(["Year", "months", "mean VoC position / its 1931-2020 mean", "months long",
                    "buy-and-hold return (sum, standardised units)", "VoC return (sum, in units of its mean absolute position)"], rows)
    return tab, facts


def no_negative_zero(text: str) -> str:
    """Print '-0.00' (a rounded tiny negative) as '0.00'."""
    import re

    return re.sub(r"-0\.(0+)(?![0-9])", r"0.\1", text)
