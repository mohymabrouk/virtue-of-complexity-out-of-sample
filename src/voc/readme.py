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
    ctx = dict(diag=diag, chk=chk, s_p=s_p, s_q=s_q, vb=vb, n_ctry=n_ctry, pooled=pooled, n_post=n_post, years_post=years_post,
               se_post=se_post, power=power, mde=mde, first_run=first_run, countries=countries, jf=jf, jf_bh=jf_bh,
               meta_b=meta_b, tb=tb)
    text = build_text(ctx)
    (config.ROOT / "README.md").write_text(text, encoding="utf-8")


def build_text(x) -> str:
    from voc import readme_text

    return readme_text.render(x)
