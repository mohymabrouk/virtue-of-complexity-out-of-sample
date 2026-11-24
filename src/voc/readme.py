"""README.md, assembled from results/ only (numbers are read from the result files, never typed in)."""

import json

import numpy as np
import pandas as pd

from voc import config, metrics, pipeline, report

R = config.RESULTS
TARGET = report.TARGET


def _load():
    chk = json.loads((R / "replication_check.json").read_text())
    paper, post = report.load_partA()
    log = [json.loads(x) for x in (R / "test_log.jsonl").read_text().splitlines() if x.strip()]
    done = [e for e in log if e["event"] == "done"]
    meta_b = json.loads((R / "partB_meta.json").read_text())
    sel = pd.read_csv(R / "partB_country_selection.csv")
    return chk, paper, post, done, meta_b, sel


def _row(t, conv, period, name):
    return t[(t.conv == conv) & (t.period == period) & (t.strategy == name)].iloc[0]


def write() -> None:
    chk, paper, post, done, meta_b, sel = _load()
    first_run = done[0]
    n_post = first_run["n_months"]
    years_post = n_post / 12
    se_post = float(np.sqrt(12 / n_post))  # annualised Sharpe SE for an iid normal series with small SR
    power = metrics.power_one_sided(config.PAPER_TABLE1["nonlinear_z1e3"]["IR"], years_post)
    mde = (1.645 + 0.842) / np.sqrt(years_post)

    s_p = {c: {n: _row(paper, c, "paper", n) for n in pipeline.main_strategy_names()} for c in config.CONVENTIONS}
    s_q = {c: {n: _row(post, c, "post", n) for n in pipeline.main_strategy_names()} for c in config.CONVENTIONS}
    tb = report.partB_frames()
    nonus = tb[tb.location != "usa"]
    n_ctry = nonus.location.nunique()
    vb = {c: nonus[(nonus.conv == c) & (nonus.strategy == TARGET)] for c in config.CONVENTIONS}
    pooled = pd.read_csv(R / "partB_pooled_table.csv").set_index(["conv", "strategy"])
    countries = sorted(set(nonus.location))
    jf = {c: _row(post, c, "post_jf", TARGET) for c in config.CONVENTIONS}
    jf_bh = {c: _row(post, c, "post_jf", "Buy-and-hold") for c in config.CONVENTIONS}

    d = json.loads((R / "replication_diagnostics.json").read_text())
    la = d["std_window_includes_forecast_month (look-ahead)"]
    diag = {"bh_std": s_p["std"]["Buy-and-hold"].sharpe, "bh_raw": s_p["raw"]["Buy-and-hold"].sharpe,
            "lag_sr": d["std_trailing_t-12..t-1"]["voc_z1e3"]["sharpe"], "rms_sr": d["rms_trailing_t-11..t"]["voc_z1e3"]["sharpe"],
            "la_sr": la["voc_z1e3"]["sharpe"], "la_t": la["voc_z1e3"]["alpha_t"], "la_lin0": la["linear_ridgeless_sharpe"],
            "la_lin_t": la["linear_z1e3"]["alpha_t"], "la_lin_sr": la["linear_z1e3"]["sharpe"],
            "vm_t": d["nagel_volmom_raw"]["alpha_t"], "vm_ir": d["nagel_volmom_raw"]["ir"]}
    ext = json.loads((R / "external_replication_check.json").read_text())
    diag.update(ext_sr=ext["voc_sharpe_mean_over_draws"], ext_mkt=ext["market_sharpe"], ext_n=ext["n_draws"],
                ext_commit=ext["commit"], ext_repo=ext["repo"])
    # second external check (PLAN.md 14-15) and construction B (PLAN.md 16)
    e2 = json.loads((R / "external_data_diagnostic.json").read_text())
    er = e2["runs"]["ext data (its X, its R)"]
    diag.update(e2_repo=e2["source"]["repo"], e2_commit=e2["source"]["commit"], e2_n=e2["n_seeds"],
                e2_sr=er["voc_z1e3"]["sharpe"], e2_t=er["voc_z1e3"]["alpha_t"], e2_ir=er["voc_z1e3"]["ir"],
                e2_lin_sr=er["linear_z1e3"]["sharpe"], e2_lin_t=er["linear_z1e3"]["alpha_t"],
                e2_lin0=er["linear_ridgeless_sharpe"],
                hyb_ret=e2["runs"]["our X, its R"]["voc_z1e3"]["sharpe"],
                hyb_pred=e2["runs"]["its X, our R"]["voc_z1e3"]["sharpe"])
    cb = json.loads((R / "construction_b.json").read_text())
    for i, v in enumerate(cb["decomposition_paper_sample"].values()):
        diag[f"dec{i}_sr"], diag[f"dec{i}_t"] = v["voc_z1e3_sharpe"], v["voc_z1e3_alpha_t"]
    for con in ("A", "B"):
        for per in ("paper", "post"):
            v = cb[con][per]
            k = f"{con.lower()}_{per}"
            diag.update({f"{k}_sr": v["voc_z1e3_per_draw"]["sharpe"], f"{k}_t": v["voc_z1e3_per_draw"]["alpha_t"],
                         f"{k}_ir": v["voc_z1e3_per_draw"]["ir"], f"{k}_ens": v["voc_z1e3_ensemble"]["sharpe"],
                         f"{k}_ens_se": v["voc_z1e3_ensemble"]["sharpe_se"],
                         f"{k}_ens_t": v["voc_z1e3_ensemble"]["alpha_t"], f"{k}_bh": v["bh_sharpe"],
                         f"{k}_lin_sr": v["linear_z1e3"]["sharpe"], f"{k}_n": v["n_months"]})
    diag.update(b_draws=cb["B"]["n_draws"], post_first=cb["B"]["post"]["first_target"],
                post_last=cb["B"]["post"]["last_target"])
    ctx = dict(diag=diag, chk=chk, s_p=s_p, s_q=s_q, vb=vb, n_ctry=n_ctry, pooled=pooled, n_post=n_post, years_post=years_post,
               se_post=se_post, power=power, mde=mde, first_run=first_run, countries=countries, jf=jf, jf_bh=jf_bh,
               meta_b=meta_b, tb=tb, dec_table=_decomp_table(cb, e2))
    text = build_text(ctx)
    (config.ROOT / "README.md").write_text(text, encoding="utf-8")


def _decomp_table(cb: dict, e2: dict) -> str:
    """Markdown table: the four data switches added one at a time (results/construction_b.json)."""
    pub = config.PAPER_TABLE1
    rows = ["| Data construction (same engine; P = 12,000, z = 10³, 1931-2020, 10 draws) | VoC Sharpe | VoC α t | VoC IR "
            "| Linear z = 10³ Sharpe | Linear α t | Buy-and-hold Sharpe |", "|---|---|---|---|---|---|---|"]
    labels = ["A: ours, pre-registered (frozen)",
              "+ (i) predictor std from each series' first month in the Goyal file",
              "+ (ii) inflation not lagged (uses CPI about two weeks before its release)",
              "+ (iii) S&P 500 return from the Goyal file instead of French Mkt-RF",
              "+ (iv) uncentred RMS return volatility instead of std = **B**"]
    for lab, v in zip(labels, cb["decomposition_paper_sample"].values()):
        rows.append(f"| {lab} | {v['voc_z1e3_sharpe']:.2f} | {v['voc_z1e3_alpha_t']:.1f} | {v['voc_z1e3_ir']:.2f} | "
                    f"{v['linear_z1e3_sharpe']:.2f} | {v['linear_z1e3_alpha_t']:.1f} | {v['bh_sharpe']:.2f} |")
    er = e2["runs"]["ext data (its X, its R)"]
    rows.append(f"| rrikrv's data file, our engine | {er['voc_z1e3']['sharpe']:.2f} | {er['voc_z1e3']['alpha_t']:.1f} | "
                f"{er['voc_z1e3']['ir']:.2f} | {er['linear_z1e3']['sharpe']:.2f} | "
                f"{er['linear_z1e3']['alpha_t']:.1f} | {er['bh_sharpe']:.2f} |")
    rows.append(f"| KMZ, Table 1 | {pub['nonlinear_z1e3']['SR']:.2f} | {pub['nonlinear_z1e3']['IR_t']:.1f} | "
                f"{pub['nonlinear_z1e3']['IR']:.2f} | {pub['linear_z1e3']['SR']:.2f} | "
                f"{pub['linear_z1e3']['IR_t']:.1f} | – |")
    return "\n".join(rows)


def build_text(x) -> str:
    from voc import readme_text

    return readme_text.render(x)
