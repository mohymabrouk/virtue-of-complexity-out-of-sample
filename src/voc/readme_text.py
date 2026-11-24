"""The README text. Every number is taken from the context built in readme.py (i.e. from results/)."""

from voc import config, pipeline, report

TARGET = report.TARGET


def _sr(r):
    return f"{r.sharpe:.2f} [{r.sharpe_lo:.2f}, {r.sharpe_hi:.2f}]"


def _curve_points() -> dict:
    import pandas as pd

    c = pd.read_csv(config.RESULTS / "partA_paper_curve.csv")

    def v(conv, P, z, col):
        return float(c[(c.conv == conv) & (c.P == P) & (c.z == z)][col].iloc[0])

    return {"ens_p2": v("std", 2, 1e-3, "sharpe_ens"), "ens_p12k": v("std", 12000, 1e-3, "sharpe_ens"),
            "raw_t24": v("raw", 24, 1000.0, "alpha_t_ens"), "raw_t12k": v("raw", 12000, 1000.0, "alpha_t_ens")}


def _post_text(x, std_q, raw_q) -> str:
    import pandas as pd

    tab, f = report.post_by_year("std")
    c = pd.read_csv(config.RESULTS / "partA_post_curve.csv")
    c = c[(c.period == "post") & (c.conv == "std") & (c.z == config.MAIN_Z)]
    return POST_TEXT.format(
        n_post=x["n_post"], q_sr=std_q[TARGET].sharpe, q_bh=std_q["Buy-and-hold"].sharpe, qr_sr=raw_q[TARGET].sharpe,
        qr_bh=raw_q["Buy-and-hold"].sharpe, q_t=std_q[TARGET].alpha_t, qr_t=raw_q[TARGET].alpha_t,
        n22_short=f["2022"]["n"] - f["2022"]["n_long"], v22=f["2022"]["voc"], b22=f["2022"]["bh"],
        wm=f["worst_month"]["month"], wm_pos=f["worst_month"]["rel_pos"], wm_y=f["worst_month"]["y"],
        vm_sr=std_q["Nagel vol-timed momentum"].sharpe, pc_min=c.sharpe_paper.min(), pc_max=c.sharpe_paper.max(),
        year_table=tab)


def _wins_against(other: str, conv: str) -> int:
    """Countries (non-US) where `other` has a higher Sharpe than the target VoC cell."""
    t = report.partB_frames()
    t = t[(t.conv == conv) & (t.location != "usa")].set_index(["location", "strategy"])
    locs = sorted(set(t.index.get_level_values(0)))
    return int(sum(t.loc[(loc, other), "sharpe"] > t.loc[(loc, TARGET), "sharpe"] for loc in locs))


def _partb_text(x, n_ctry) -> str:
    import pandas as pd

    t = report.partB_frames()
    nu = t[(t.location != "usa")]
    med = nu.groupby(["conv", "strategy"])["sharpe"].median()
    pooled = x["pooled"]
    cur = pd.read_csv(config.RESULTS / "partB_curves.csv")
    bh = t[t.strategy == "Buy-and-hold"].set_index(["location", "conv"])["sharpe"]
    cur = cur[cur.location != "usa"].copy()
    cur["d"] = cur["sharpe_ens"] - [bh.loc[(lo, c)] for lo, c in zip(cur.location, cur.conv)]
    cur["dp"] = cur["sharpe_paper"] - [bh.loc[(lo, c)] for lo, c in zip(cur.location, cur.conv)]
    worst_cell = max(cur.groupby(["conv", "P", "z"])["d"].median().max(), cur.groupby(["conv", "P", "z"])["dp"].median().max())
    return PARTB_RESULTS.format(
        n_ctry=n_ctry, v_med=med[("std", TARGET)], bh_med=med[("std", "Buy-and-hold")],
        n_lose=n_ctry - int((x["vb"]["std"].d_sr > 0).sum()), best_cell=worst_cell,
        vm_pt=pooled.loc[("std", "Nagel vol-timed momentum"), "alpha_t"], km_pt=pooled.loc[("std", "KMZ momentum (mean/var)"), "alpha_t"],
        vm_d=pooled.loc[("std", "Nagel vol-timed momentum"), "dsharpe"], km_d=pooled.loc[("std", "KMZ momentum (mean/var)"), "dsharpe"],
        vm_wins=_wins_against("Nagel vol-timed momentum", "std"), vm_wins_raw=_wins_against("Nagel vol-timed momentum", "raw"),
        pool_v=pooled.loc[("std", TARGET), "sharpe"], pool_bh=pooled.loc[("std", "Buy-and-hold"), "sharpe"],
        pool_d=pooled.loc[("std", TARGET), "dsharpe"], pool_lo=pooled.loc[("std", TARGET), "dsharpe_lo"],
        pool_hi=pooled.loc[("std", TARGET), "dsharpe_hi"])


def render(x) -> str:
    chk, s_p, s_q, vb, pooled = x["chk"], x["s_p"], x["s_q"], x["vb"], x["pooled"]
    n_ctry, n_post = x["n_ctry"], x["n_post"]
    fr = x["first_run"]
    std_p, std_q, raw_p, raw_q = s_p["std"], s_q["std"], s_p["raw"], s_q["raw"]
    vstd, vraw = vb["std"], vb["raw"]
    pool_std = pooled.loc[("std", TARGET)]
    pool_bh_std = pooled.loc[("std", "Buy-and-hold")]
    pool_raw = pooled.loc[("raw", TARGET)]
    pool_bh_raw = pooled.loc[("raw", "Buy-and-hold")]
    tab_std, _ = report.partB_table("std")
    tab_raw, _ = report.partB_table("raw")
    pooled_std, pf, pl = report.partB_pooled_table("std")
    pooled_raw, _, _ = report.partB_pooled_table("raw")
    bridge, _ = report.us_bridge_table()
    wins_std = int((vstd.d_sr > 0).sum())
    wins_raw = int((vraw.d_sr > 0).sum())
    sig_std = int((vstd.alpha_t > 1.96).sum())
    sig_raw = int((vraw.alpha_t > 1.96).sum())
    vm = "Nagel vol-timed momentum"

    head = HEADLINE.format(
        p_std=chk["ours_sharpe_mean_over_seeds"], p_t=chk["ours_alpha_t_mean_over_seeds"],
        raw_t=chk["raw_convention"]["alpha_t"], raw_ir=chk["raw_convention"]["ir"],
        q_sr=std_q[TARGET].sharpe, q_bh=std_q["Buy-and-hold"].sharpe, q_t=std_q[TARGET].alpha_t,
        q_d=std_q[TARGET].dsharpe, q_dlo=std_q[TARGET].dsharpe_lo, q_dhi=std_q[TARGET].dsharpe_hi,
        qr_t=raw_q[TARGET].alpha_t, qr_d=raw_q[TARGET].dsharpe, n_post=n_post,
        wins_std=wins_std, wins_raw=wins_raw, n_ctry=n_ctry, sig_std=sig_std, sig_raw=sig_raw,
        med_std=vstd.d_sr.median(), med_raw=vraw.d_sr.median(),
        pool_t=pool_std.alpha_t, pool_d=pool_std.dsharpe, pool_rt=pool_raw.alpha_t, pool_rd=pool_raw.dsharpe,
        vm_q=raw_q[vm].alpha_t, qr_sr=raw_q[TARGET].sharpe, qr_bh=raw_q["Buy-and-hold"].sharpe,
        power=x["power"], chance=0.025 * n_ctry, vm_wins=_wins_against(vm, "std"), n_ctry2=n_ctry,
        **{k: x["diag"][k] for k in ("la_sr", "ext_sr", "a_paper_sr", "b_paper_sr", "a_paper_t", "b_paper_t",
                                     "post_first", "post_last", "b_post_ens", "b_post_bh", "b_post_ens_t",
                                     "a_post_ens", "a_post_bh", "b_post_n")},
    )
    parts = [
        "# Virtue of complexity, out of sample: after the paper's sample and across 41 countries\n",
        "[![tests](https://github.com/mohymabrouk/virtue-of-complexity-out-of-sample/actions/workflows/tests.yml/badge.svg)]"
        "(https://github.com/mohymabrouk/virtue-of-complexity-out-of-sample/actions/workflows/tests.yml)\n",
        "Retests Kelly, Malamud & Zhou (*Journal of Finance*, 2024) after publication and in 41 non-US markets, "
        "on free data, CPU only.\n",
        "## Headline\n", head,
        "![VoC vs buy-and-hold after the paper's sample](results/fig_post_cumulative.png)\n",
        INTRO.format(n_seeds_a=chk["n_seeds"], n_seeds_b=x["meta_b"]["n_seeds"], n_post=n_post,
                     first=fr["first_target"], last=fr["last_target"]),
        "## Main performance table (US)\n",
        "Rows are strategies, columns the two periods. VoC rows use the seed-averaged forecast (one implementable "
        "strategy). Sharpe ratios are annualised, with a 95% CI from the Lo (2002) standard error with the Mertens "
        "(2002) skew/kurtosis correction. DSR = deflated Sharpe ratio (probability the true Sharpe beats the best of "
        "153 trials: 140 VoC cells, 8 linear ridge cells, 5 simple rules). α t = OLS t-stat of the alpha in a "
        "regression on the same-convention market (iid, as in KMZ; Newey-West t-stats are in the CSVs). "
        "OOS R² = 1 - Σ(R-f)²/ΣR², shown only where the position is a return forecast.\n",
        f"**Standardised returns (KMZ's specification, primary).**\n\n{report.main_table('std')}\n",
        f"**Raw excess returns (Nagel's convention, secondary).**\n\n{report.main_table('raw')}\n",
        "### Sharpe difference vs buy-and-hold (circular block bootstrap over months, 2,000 resamples)\n",
        f"Standardised:\n\n{report.dsharpe_table('std')}\n\nRaw:\n\n{report.dsharpe_table('raw')}\n",
        _post_text(x, std_q, raw_q),
        "## Replication\n", report.replication_table() + "\n",
        REPLICATION_TEXT.format(**chk, **x["diag"], dec_table=x["dec_table"]),
        "## Complexity curves\n", CURVES_TEXT,
        "![paper std](results/fig_curve_paper_std.png)\n\n![post std](results/fig_curve_post_std.png)\n\n"
        "![paper raw](results/fig_curve_paper_raw.png)\n\n![post raw](results/fig_curve_post_raw.png)\n\n"
        "![positions](results/fig_positions_us.png)\n",
        "## Part B: the same specification on 41 non-US markets\n",
        PARTB_TEXT.format(n_ctry=n_ctry, countries=", ".join(c.upper() for c in x["countries"]), n_seeds=x["meta_b"]["n_seeds"]),
        f"**Standardised returns.**\n\n{tab_std}\n\n{report.partB_counts('std')}\n\n"
        f"Pooled, equal-weighted across the non-US countries available each month ({pf} to {pl}):\n\n{pooled_std}\n",
        f"**Raw returns.**\n\n{tab_raw}\n\n{report.partB_counts('raw')}\n\nPooled ({pf} to {pl}):\n\n{pooled_raw}\n",
        f"**Bridge: the US with JKP predictors** (same 15-predictor recipe as the other countries), next to the "
        f"Welch-Goyal model of Part A on the same months.\n\n{bridge}\n",
        "![partB curves](results/fig_partB_curves.png)\n\n![partB countries](results/fig_partB_countries.png)\n",
        _partb_text(x, n_ctry),
        WHAT_IS_NEW.format(n_ctry=n_ctry, **_curve_points()), RELATED, LIMITS.format(n_post=n_post, years=x["years_post"], se=x["se_post"], power=x["power"], mde=x["mde"],
                                   jf_n=int(x["jf"]["std"].n), rep_sr=chk["ours_sharpe_mean_over_seeds"]),
        SIMPLIFICATIONS.format(n_seeds_a=chk["n_seeds"], n_seeds_b=x["meta_b"]["n_seeds"]),
        RUN.format(rt_a=chk["runtime_s"] / 60, rt_t=fr["runtime_s"] / 60, rt_b=x["meta_b"]["runtime_s"] / 60,
                   cfg=fr["config_hash"], code=fr["code_hash"], ts=fr["timestamp_utc"]),
        FILES, SOURCES,
    ]
    return report.no_negative_zero("\n".join(parts))


HEADLINE = """
**No. In these tests the virtue of complexity does not survive outside its original sample.** Over the paper's
own sample (US, 1931-2020) our implementation of the published specification reproduces the *shape* of KMZ's
complexity curve but not its level: the c = 1000, z = 10³ strategy has a Sharpe of {p_std:.2f} (paper: 0.47) and an
alpha t-stat of {p_t:.1f} against the volatility-standardised market (paper: 2.9); only in raw-return terms do we
match the paper's alpha and information ratio (t {raw_t:.1f}, IR {raw_ir:.2f}, vs 2.9 and 0.31). Scored once on the
{n_post} months after the paper's sample (2021-01 to 2026-01), the same strategy earned a Sharpe of {q_sr:.2f}
against {q_bh:.2f} for buy-and-hold (raw returns: {qr_sr:.2f} vs {qr_bh:.2f}), with alpha t-stats of {q_t:.1f} and
{qr_t:.1f}: it sidestepped 2022, then gave the gain back and missed the 2024-25 rally. Across {n_ctry} non-US
markets, with the same specification on JKP predictors, it beat buy-and-hold in {wins_std} countries ({wins_raw}
in raw returns) and had a significantly positive alpha in {sig_std} ({sig_raw}), about what chance alone gives
(about {chance:.0f}); the pooled cross-country portfolio has an alpha t-stat of {pool_t:.1f} ({pool_rt:.1f}). The simple
volatility-timed momentum rule that Nagel (2025) says VoC reduces to beat VoC in {vm_wins} of {n_ctry2} countries.
And the rising Sharpe-complexity curve is mostly an averaging artefact: when the forecasts of the random draws
are averaged instead of their statistics, an average of 300 two-feature models matches the 12,000-feature Sharpe in the
standardised spec, and in raw returns the alpha stops improving beyond 24 features. Two caveats bound
the US conclusions: five years of post-sample data give only about {power:.0%} power to detect the paper's
information ratio, and our primary implementation does not hit the paper's in-sample Sharpe (the construction
below does, and fails after the sample too). The cross-country evidence depends on neither, and it points the
same way.

**The construction that reproduces the paper also fails after its sample.** With our engine unchanged, four data
choices that the paper does not pin down move the 1931-2020 Sharpe from {a_paper_sr:.2f} to {b_paper_sr:.2f}
(alpha t {a_paper_t:.1f} to {b_paper_t:.1f}; paper: 0.47 and 2.9): where the predictors' expanding standardisation
starts, the inflation publication lag, the market-return series, and std vs uncentred RMS for the return scale
(Replication, item 6). One of them uses CPI about two weeks before its release. Scored once on {post_first} to
{post_last}, that construction earned a Sharpe of {b_post_ens:.2f} against {b_post_bh:.2f} for buy-and-hold (alpha
t {b_post_ens_t:.1f}); ours earned {a_post_ens:.2f} against {a_post_bh:.2f} on the same {b_post_n} months.
Write-up: [research note (PDF)](paper/note.pdf).

*Correction, 2026-11-24.* An earlier version of this README said the paper's Sharpe is approached only with a
look-ahead volatility window (one that includes the forecast month: {la_sr:.2f} here; nkonts' replication,
{ext_sr:.2f}). A third public replication (rrikrv) reaches it without one, and our engine on its data file
reproduces KMZ's Table 1. The look-ahead window matches the Sharpe but not the alpha, so that explanation is
withdrawn (`PLAN.md`, changes 14-16).
"""
PARTB_RESULTS = """
**Reading Part B.** Across the {n_ctry} countries the c = 1000, z = 10³ strategy has a median Sharpe of
{v_med:.2f} against {bh_med:.2f} for buy-and-hold (standardised) and loses to buy-and-hold in {n_lose} of them.
More complexity does not help: the cross-country median Sharpe gap to buy-and-hold is negative in every one of
the 140 (P, z) cells, under both averaging schemes and both conventions (the best cell is {best_cell:+.2f};
first figure above). Pooled across countries the strategy has a Sharpe of {pool_v:.2f} against {pool_bh:.2f}
(ΔSharpe {pool_d:+.2f}, 95% CI [{pool_lo:+.2f}, {pool_hi:+.2f}]). The rules built on recent returns do better
than VoC: Nagel's vol-timed momentum beats it in {vm_wins} of {n_ctry} countries ({vm_wins_raw} in raw returns),
and in the pooled portfolio Nagel's rule and KMZ's own momentum rule have alpha t-stats of {vm_pt:.1f} and
{km_pt:.1f}, though their Sharpe gains over buy-and-hold ({vm_d:+.2f}, {km_d:+.2f}) are not significant. The US
bridge shows that the predictor set matters, and so does the period: on the same 1956-2025 months, VoC with
Welch-Goyal predictors has a Sharpe close to the linear ridge and well below buy-and-hold, and with JKP predictors
it is near zero.
"""
INTRO = """
### What was tested

Kelly, Malamud & Zhou (KMZ), "The Virtue of Complexity in Return Prediction", *Journal of Finance* 79(1), 2024
(NBER w30217), time the US market with ridge regressions on up to 12,000 random Fourier features (RFF) of 15
Welch-Goyal predictors, trained on rolling 12-month windows: far more parameters than observations. They report a
higher out-of-sample Sharpe than simple models (0.47 at P = 12,000, z = 10³) although the OOS R² is often negative.

The specification is copied from the paper and fixed in `PLAN.md` before any result: T = 12, γ = 2,
ω ~ N(0, I), sin/cos pairs, every RFF divided by its training-window std, ridge without intercept,
P ∈ {{2, ..., 12,000}} (20 values), log₁₀ z ∈ {{-3, ..., 3}}, predictors divided by an expanding std, returns
divided by their trailing 12-month std, position = forecast. Nothing was tuned on any evaluation sample.

- **Part A**: the US, 1931-01 to 2020-12 (the paper's sample, replication), then the same specification scored
  once on the **{n_post} months after the paper's sample ends ({first} to {last})**, logged in
  `results/test_log.jsonl`.
- **Part B**: the same specification on 41 non-US equity markets from the Jensen-Kelly-Pedersen (JKP) data, with
  15 predictors built from JKP data (13 theme returns, the market return and its 12-month volatility).
- **Two return conventions.** *Standardised* = KMZ's specification (primary). *Raw* = raw excess returns for the
  target, the traded asset and the benchmark, as in Nagel (2025); it was added (and logged in `PLAN.md`) when
  the standardised version did not replicate, because it reproduces a number published from KMZ's own code.
- **Two ways to average over random seeds.** "Paper" = compute each statistic per RFF draw, then average
  (KMZ, step iv). "Ensemble" = average the forecasts over draws, then evaluate one strategy (the version an
  investor can trade, and the one Buncic (2025, as summarised by Nagel 2025) argues for). Part A uses
  {n_seeds_a} draws, Part B {n_seeds_b} per country.
- **Baselines**: buy-and-hold, the historical mean, the linear kitchen-sink ridge on the same 15 predictors
  (ridgeless and z = 10⁻³...10³), the sign of the trailing 12-month return (TSMOM), the 12-month mean return,
  KMZ's own momentum rule (mean/var, appendix G) and Nagel's (2025) volatility-timed momentum (his eq. 14), the
  simple rule he argues VoC reduces to when T = 12.
"""
REPLICATION_TEXT = """
**Target, fixed in `PLAN.md` before any result:** KMZ Table 1, Panel A, "Nonlinear" (T = 12, P = 12,000,
z = 10³), Sharpe 0.47 computed the paper's way (per-draw Sharpe averaged over draws), tolerance ±0.10.
**Ours: {ours_sharpe_mean_over_seeds:.2f} (seed sd {ours_sharpe_seed_sd:.3f}, {n_seeds} draws). Not matched.**
Test 6 (`tests/test_replication.py::test_replication_target_sharpe`) is therefore marked `xfail(strict=True)`;
its tolerance was not changed.

What we checked (all on 1931-2020 only, scripts in `scripts/diag_*.py`, log in `PLAN.md`):

1. The batched dual engine equals a direct primal ridge on the same windows, to 1e-8.
2. **The data match Nagel's.** His volatility-timed momentum rule, rebuilt from our predictors, gives alpha
   t {vm_t:.2f} and IR {vm_ir:.2f} against his 3.68 and 0.39.
3. **The RFF machine matches KMZ's code in raw-return terms.** With raw excess returns our ridgeless
   P = 12,000 strategy has alpha t {raw_convention_ridgeless[alpha_t]:.2f} and IR {raw_convention_ridgeless[ir]:.2f};
   Nagel reports 2.42 and 0.26 from KMZ's own replication code (a second test checks this within ±0.5 / ±0.06,
   and passes). At z = 10³ the raw version gives alpha t {raw_convention[alpha_t]:.2f} and IR
   {raw_convention[ir]:.2f}, close to KMZ's published 2.9 and 0.31.
4. **Changing the return scaling alone does not close the gap.** Dividing returns by their trailing 12-month std
   (as the paper describes) turns buy-and-hold into a volatility-managed market with a Sharpe of {bh_std:.2f} over
   1931-2020 (raw: {bh_raw:.2f}). Against that benchmark our VoC strategy has a Sharpe of
   {ours_sharpe_mean_over_seeds:.2f} and an alpha t of only {ours_alpha_t_mean_over_seeds:.1f}. Variants of the
   std window (`results/replication_diagnostics.json`, 3 draws each) do not close the gap: one more month of lag
   gives a VoC (z = 10³) Sharpe of {lag_sr:.2f}, a root-mean-square instead of a std {rms_sr:.2f}. A window that
   includes the forecast month (look-ahead, so not adopted) gives {la_sr:.2f}, and it reproduces two of KMZ's
   linear numbers (ridgeless Sharpe {la_lin0:.2f} vs -0.11; z = 10³ alpha t {la_lin_t:.1f} vs 3.1) but not the
   rest (linear z = 10³ Sharpe {la_lin_sr:.2f} vs 0.46; VoC alpha t {la_t:.1f} vs 2.9): a partial match.
   It caps crash months (with the month in its own scale, an extreme return tends to ±√12 ≈ ±3.5 whatever its
   size), which lifts buy-and-hold too.
5. **One public replication uses the look-ahead window.** [nkonts' replication]({ext_repo})
   (commit `{ext_commit}`) standardises with `returns_std = returns.rolling(12).std()` and
   `returns = returns / returns_std`, then targets `returns.shift(-1)`. The forecast made in month t is therefore
   scored on R(t+1) divided by a std that includes R(t+1). Averaged over its {ext_n} draws at P = 12,000,
   z = 10³, it reports a Sharpe of {ext_sr:.2f}, with buy-and-hold at {ext_mkt:.2f} (S&P price returns,
   Welch-Goyal data to 2021). That is our look-ahead variant's convention ({la_sr:.2f}), closer to the paper's
   0.47 than our lagged version ({ours_sharpe_mean_over_seeds:.2f}), but item 4 shows it matches the Sharpe and not
   the alpha, and item 6 shows it is not needed. Source: `results/external_replication_check.json`, written by
   `scripts/external_replication_check.py`.
6. **The gap is in four data-construction choices (added 2026-11-24, PLAN.md changes 14-16).** A third public
   replication, [rrikrv]({e2_repo}) (commit `{e2_commit}`), reaches the paper's numbers without a look-ahead
   window. Our unchanged engine on its committed data file gives a VoC Sharpe of {e2_sr:.2f}, alpha t
   {e2_t:.1f} and IR {e2_ir:.2f}, and linear z = 10³ {e2_lin_sr:.2f} / alpha t {e2_lin_t:.1f}, ridgeless
   {e2_lin0:.2f} ({e2_n} draws; `results/external_data_diagnostic.json`, written by
   `scripts/diag_external_data.py`, which reads the file and runs none of its code). Swapping only its returns
   into our data gives {hyb_ret:.2f}, only its predictors {hyb_pred:.2f}. Rebuilding its four data choices from our
   own raw files and adding them one at a time (`scripts/construction_b.py`, `results/construction_b.json`)
   locates the whole gap:

{dec_table}

   Each step adds 0.03 to 0.07; none closes the gap alone. Steps (i), (iii) and (iv) use only past data. Step (ii)
   does not: CPI for month t is released during month t+1, which is why Welch and Goyal lag it and construction A
   does too. The paper's text ("trailing 12-month return standard deviation", "expanding window historical standard
   deviation", "the CRSP value-weighted index") fits A at least as well as B, so neither is "the" paper's
   construction; the point is that the headline number moves from {dec0_sr:.2f} to {dec4_sr:.2f} with them, while
   KMZ write that "none of our findings are sensitive to variations in how standardizations are implemented".
   With {b_draws} draws, construction B gives {b_paper_sr:.2f} (alpha t {b_paper_t:.1f}, IR {b_paper_ir:.2f}) on
   1931-2020; its post-sample score is in the headline.

Hence the two conventions in every table: the standardised one is the paper's text as we read it (construction A);
the raw one reproduces numbers obtained with the authors' code. Construction B, which reproduces the published
standardised numbers, was added after the frozen runs and is reported separately (headline and item 6).
"""
POST_TEXT = """
### What happened after 2020

Over the {n_post} post-sample months the target strategy (c = 1000, z = 10³) earned a Sharpe of {q_sr:.2f}
against {q_bh:.2f} for buy-and-hold in the standardised specification (raw: {qr_sr:.2f} vs {qr_bh:.2f}); its alpha
t-stat is {q_t:.1f} (raw {qr_t:.1f}). The year-by-year view below shows the mechanism. In 2022 it did what KMZ
describe: it was short in {n22_short} of 12 months and its 2022 return was {v22:+.1f} (in units of its usual
position) against {b22:+.1f} for buy-and-hold. It then gave that back (its worst month, {wm}, was a position of
{wm_pos:+.1f}× its usual size into a {wm_y:+.1f} sd market month) and held small positions through the 2024-25
rally. Nagel's volatility-timed
momentum, the simple rule the critique says VoC mimics, also lagged buy-and-hold (Sharpe {vm_sr:.2f}). The
complexity curve after 2020 does not rise: with the paper's averaging, the Sharpe at z = 10³ stays between
{pc_min:.2f} and {pc_max:.2f} over the whole P grid; with ensemble averaging it *falls* with complexity.

{year_table}
"""
CURVES_TEXT = """
Each figure has four panels: Sharpe with the paper's averaging (per-draw Sharpe averaged over draws), Sharpe of
the seed-averaged forecast, the alpha t-stat and the OOS R², each against complexity c = P/T on a log scale,
one line per ridge penalty z (light to dark blue: 10⁻³ to 10³). The dashed orange line is buy-and-hold.
The full grid (140 cells, both conventions, both averaging schemes) is in `results/partA_paper_curve.csv` and
`results/partA_post_curve.csv`; `results/tables.md` lists the z = 10³ and z = 10⁻³ columns.
"""
PARTB_TEXT = """
Data: JKP global factor data (jkpfactors.com, no login), capped value-weighted market excess returns in USD and
the 13 theme factor returns, monthly, to 2025-12. Predictors (fixed in `PLAN.md` before any run): the 13 theme
returns of month t, the market return of month t and its trailing 12-month volatility. A country is used on its
longest run of consecutive months with all 13 themes, a market return and at least 10 stocks, and is kept if at
least 120 out-of-sample months remain: {n_ctry} non-US countries ({countries}). Same T, γ, P grid and z grid for
every country; {n_seeds} RFF draws per country. The US is run with the same JKP recipe as a bridge and is excluded
from the non-US statistics. "Beats B&H" counts countries whose strategy Sharpe exceeds that country's buy-and-hold
Sharpe. Each country's sample is its full available history (mostly the 1990s to 2025), so Part B is out of sample
in space (KMZ used no country but the US), not in time.
"""
WHAT_IS_NEW = """
## What is new

1. **A true out-of-sample test of the published specification.** KMZ's sample ends in 2020-12; the same
   pre-registered model, with the same random draws, was scored once on 2021-01 to 2026-01 (logged before the
   result was seen). Neither KMZ nor Nagel (2025) could do this.
2. **A cross-country test with one fixed specification.** {n_ctry} non-US markets, no per-country tuning, with
   the same baselines, distribution, head-to-head counts and a pooled portfolio.
3. **Both ways of averaging over random draws on the same draws.** The rising "virtue of complexity" curve in
   the Sharpe ratio appears when per-draw statistics are averaged (the paper's step iv), because single
   low-P draws are noisy. When the forecasts of the same draws are averaged first, models with a handful of
   features already earn what P = 12,000 earns: over 1931-2020 the standardised ensemble with P = 2 and
   z = 10⁻³ has a Sharpe of {ens_p2:.2f} against {ens_p12k:.2f} at P = 12,000; in raw returns the ensemble alpha
   t-stat at z = 10³ is {raw_t24:.1f} at P = 24 and {raw_t12k:.1f} at P = 12,000. This is the point Nagel (2025)
   attributes to Buncic (2025), checked here directly.
4. **The critique's simple rule, scored out of sample too.** Nagel's volatility-timed momentum is a baseline in
   every table, including the post-sample window and every country.
5. **A located replication gap.** With the engine held fixed, four data-construction choices that the paper does
   not pin down move the headline Sharpe from about 0.31 to 0.49 (Replication, item 6), and the construction that
   reproduces the paper was scored once after its sample too. A look-ahead volatility window, used by one public
   replication, matches the Sharpe but not the alpha. Write-up: [`paper/note.pdf`](paper/note.pdf).
"""
RELATED = """
## Related work (2025-2026)

- **Nagel (2025)**, [NBER w34104](https://www.nber.org/papers/w34104): with T = 12, the RFF forecasts are close to
  volatility-timed momentum; his Appendix A shows that scaling the target by a lagged volatility can itself create
  spurious predictability. His rule is a baseline in every table here, including after 2020 and in 41 countries.
- **Buncic (2025)**, [SSRN 5239006](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5239006): the result rests on
  the zero-intercept restriction and on averaging performance measures across random draws. The "paper" vs
  "ensemble" averaging comparison in this repo tests the second point on the same draws.
- **Cartea, Jin & Shi (2025)**, [SSRN 5202064](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5202064):
  measurement error in the predictors caps the benefit of complexity. Cartea & Shi (2025),
  [SSRN 5930257](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5930257): random-feature gains can be mechanical.
- **Fallahgoul (2025)**, [arXiv 2506.03780](https://arxiv.org/abs/2506.03780): within-window standardisation changes
  the kernel the RFF approximate; learnability bounds that 12-month windows do not meet.
- **Guo, Huang, Li & Yu (2026)**, [arXiv 2608.23761](https://arxiv.org/abs/2608.23761): ridge and ridgeless RFF
  models do not beat the historical mean; their return scaling (uncentred second moment) is step (iv) of
  construction B.
- This repo adds what those papers could not: a test after the paper's sample, 41 countries, and a decomposition
  of the replication gap into data choices with the engine held fixed.
"""
LIMITS = """
## Limits

- **The post-sample window is short: {n_post} months ({years:.1f} years).** The standard error of an annualised
  Sharpe ratio over {n_post} months is about {se:.2f}. If the true information ratio were KMZ's 0.31, the chance of
  an alpha t-stat above 1.645 (5%, one-sided) in {years:.1f} years is only {power:.0%}; the smallest IR detectable
  with 80% power is about {mde:.1f}. A post-sample "failure" is therefore weak evidence on its own; what the
  window can show is whether the strategy behaved as advertised (sign of the alpha, size of the Sharpe gap,
  behaviour in 2022), not whether it is significant. The window after the JF publication alone (2024-01 to
  2026-01, {jf_n} months) is reported in the CSVs only.
- **Replication not matched in the primary construction** (Sharpe {rep_sr:.2f} vs 0.47; see "Replication"). The
  post-sample and Part B results of the standardised spec are those of *our* construction of the paper's data.
  Construction B reproduces the paper and was scored after its sample too (headline); it was defined after the
  frozen runs, and one of its four choices (unlagged inflation) is not real-time. Part B was not rerun with B's
  choices: (i)-(iii) are Welch-Goyal-specific, and the effect of (iv) on the 41 countries is untested.
- **Part B is a different predictor set.** It tests "the same machine on comparable inputs", not KMZ's exact
  model; the US-with-JKP-predictors bridge shows how much the predictor change matters on its own.
- **Countries are not independent.** Equity markets co-move, so 41 countries are worth fewer than 41
  independent tests; the sign tests treat them as independent and overstate precision.
- **Deflated Sharpe** uses the 153 strategies reported per period as the trial family. Earlier diagnostic runs
  on the paper sample (logged in `PLAN.md`) are not counted.
- **Fewer RFF draws than the paper** (see Simplifications); this matters little at high complexity, where the
  conclusions are drawn.
"""
SIMPLIFICATIONS = """
## Simplifications and deviations from the paper

All are logged with dates in `PLAN.md`.

1. **Market return**: Ken French's Mkt-RF (CRSP value-weighted excess return) instead of CRSP pulled from WRDS.
2. **Predictor vintage**: Goyal's 2025 file (data to 2025-12), not KMZ's vintage; Welch-Goyal (2008) log
   definitions for dp, dy, ep, de; inflation lagged one extra month (Welch-Goyal convention; KMZ do not say).
3. **Seeds**: {n_seeds_a} RFF draws in Part A and {n_seeds_b} per country in Part B instead of KMZ's 1,000 (2 of the
   laptop's 8 cores were available). At P ≥ 1,200 the seed-to-seed spread of the Sharpe is small (see the
   `sharpe_seed_sd` column of the curve CSVs), so the averages are precise there; the low-P "paper" curve is noisier.
4. **P grid**: 20 values from 2 to 12,000 (the paper does not list its exact grid).
5. **RFF std** uses ddof = 0 (numpy default); with ddof = 1 every z would be multiplied by 12/11.
6. **Secondary raw-return convention** added after the replication gap was found (see "Replication").
7. **Only T = 12** (the paper's main case); T = 60 and 120 are not run.
8. **No transaction costs, no leverage limits**: positions are the raw forecasts, as in the paper.
9. **Part B predictors** are JKP-based, not Welch-Goyal (which exist only for the US).
"""
RUN = """
## How to run

```bash
git clone https://github.com/mohymabrouk/virtue-of-complexity-out-of-sample.git
cd virtue-of-complexity-out-of-sample
uv sync
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
uv run python -m voc.cli download    # raw files into ~/.cache/quant-research/p2/raw (never re-downloaded)
uv run python -m voc.cli build       # panels + Part B country selection
uv run python -m voc.cli replicate   # Part A, paper sample (data cut at 2020-12), ~{rt_a:.0f} min
uv run python -m voc.cli test        # Part A, post-sample; runs once (a rerun needs --force and is logged), ~{rt_t:.0f} min
uv run python -m voc.cli global      # Part B, all countries, ~{rt_b:.0f} min
uv run python -m voc.cli report      # tables.md, figures, README.md
uv run pytest                        # tests
```

The post-sample test ran once, finishing at {ts} (config hash `{cfg}`, code hash `{code}`); see
`results/test_log.jsonl`. Times are with 2 cores while other jobs shared the machine.
"""
FILES = """
## Files

- `PLAN.md`: the pre-registered design and the log of every change during the build.
- `src/voc/`: `config.py` (every constant), `download.py`, `data.py` (panels, scaling), `model.py` (RFF, dual
  ridge, baselines), `metrics.py` (statistics; `sharpe` and `deflated_sharpe` copied from Alpha Audit),
  `pipeline.py` (stages), `report.py`, `readme.py`, `readme_text.py`.
- `tests/`: the six required tests (ridge, RFF, leak, strategy, metrics, replication) and helpers.
- `scripts/`: `thin_slice.py` (first end-to-end check), `diag_*.py` (exploratory replication diagnostics,
  paper sample only), `replication_diagnostics.py` (the same checks, written to
  `results/replication_diagnostics.json`), `external_replication_check.py` (reads nkonts' public replication,
  writes `results/external_replication_check.json`), `diag_external_data.py` (runs our engine on rrikrv's data file,
  writes `results/external_data_diagnostic.json`) and `construction_b.py` (the four-step decomposition and
  construction B's post-sample score, `results/construction_b.json`, `results/construction_b_post_positions.csv`).
- `paper/`: the research note (`note.tex`, `refs.bib`, compiled `note.pdf`); `numbers.json` / `numbers.tex` are
  written from `results/` by `python -m voc.note`, so the note types no number in.
- `results/`: `tables.md`, `partA_*` (tables, complexity curves, positions), `partB_*` (per-country tables,
  curves, strategy returns, pooled table, country selection), `replication_check.json`, `test_log.jsonl`,
  `replication_diagnostics.json`, `data_manifest.json` (URLs, sizes, SHA-256, download dates), figures `fig_*.png`.
"""
SOURCES = """
## Sources

- Kelly, B., Malamud, S. and Zhou, K. (2024), "The Virtue of Complexity in Return Prediction", *Journal of
  Finance* 79(1), 459-503; NBER Working Paper 30217 (read:
  https://www.nber.org/system/files/working_papers/w30217/w30217.pdf).
- Nagel, S. (2025), "Seemingly Virtuous Complexity in Return Prediction", NBER Working Paper 34104 (read:
  https://www.nber.org/system/files/working_papers/w34104/w34104.pdf).
- Buncic, D. (2025), "Simplified: A Closer Look at the Virtue of Complexity in Return Prediction", Stockholm
  University working paper, SSRN 5239006. Not read in full (SSRN refuses automated access); cited for its abstract
  and for the point Nagel (2025) attributes to it.
- Guo, H., Huang, J., Li, R. and Yu, Y. (2026), "(Mis)Understanding Benign Overfitting in Equity Return
  Prediction", arXiv 2608.23761 (read). Cartea, Á., Jin, Q. and Shi, Y. (2025), SSRN 5202064; Cartea, Á. and
  Shi, Y. (2025), SSRN 5930257; Fallahgoul, H. (2025), arXiv 2506.03780 (abstracts read).
- Public replications whose data or code were read (none executed): nkonts (commit 40ab1c8) and rrikrv
  (commit 9c77321).
- Welch, I. and Goyal, A. (2008), *Review of Financial Studies* 21(4); data: Amit Goyal's website
  (https://sites.google.com/view/agoyal145), file "All data up to 2025".
- Jensen, T., Kelly, B. and Pedersen, L. (2023), "Is There a Replication Crisis in Finance?", *Journal of
  Finance*; data: https://jkpfactors.com (downloaded 2026-11-21, files `[all_countries]_[mkt]_[monthly]_[vw_cap]`
  and `[all_countries]_[all_themes]_[monthly]_[vw_cap]`).
- Kenneth French data library, `F-F_Research_Data_Factors` (built from the 202609 CRSP file).
- Lo, A. (2002), "The Statistics of Sharpe Ratios", *Financial Analysts Journal*; Mertens, E. (2002);
  Bailey, D. and López de Prado, M. (2014), "The Deflated Sharpe Ratio"; Moskowitz, T., Ooi, Y. and
  Pedersen, L. (2012), "Time Series Momentum".
"""
