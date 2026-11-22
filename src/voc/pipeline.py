"""Stages `replicate` (paper sample), `test` (post-sample, run once, logged) and `global` (Part B)."""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from voc import config, data, metrics, model

log = logging.getLogger(__name__)

STAT_KEYS = ("sharpe", "r2", "alpha_t", "ir", "mean", "vol")
NP, NZ = len(config.P_GRID), len(config.Z_GRID)
KT, JT = config.P_GRID.index(config.TARGET_P), config.Z_GRID.index(config.TARGET_Z)
FORECAST_STRATEGIES = {"hist_mean", "mom12"}  # simple rules whose position is a return forecast (R2 is meaningful)


def _zlabel(z: float) -> str:
    return "0+" if z == 0 else f"1e{int(round(np.log10(z)))}"


def voc_name(P: int, z: float) -> str:
    c = P / config.T
    cs = f"{c:g}" if c >= 1 else f"{c:.2f}"
    return f"VoC c={cs} (P={P}), z={_zlabel(z)}"


MAIN_VOC = [(P, config.MAIN_Z) for P in config.MAIN_P] + [(config.TARGET_P, 1e-3)]
SIMPLE_LABELS = {"hist_mean": "Historical mean", "tsmom_sign": "TSMOM sign", "mom12": "Mom12 (equal weights)",
                 "kmz_mom": "KMZ momentum (mean/var)", "nagel_volmom": "Nagel vol-timed momentum"}


# ---------------------------------------------------------------------------------------------------------------
# hashes and logging
# ---------------------------------------------------------------------------------------------------------------
def config_hash() -> str:
    items = {k: getattr(config, k) for k in dir(config) if k.isupper() and not k.startswith("_")}
    items = {k: (str(v) if not isinstance(v, (int, float, str, tuple, list, dict)) else v) for k, v in items.items()
             if k not in {"ROOT", "CACHE", "RAW", "BUILT", "RESULTS", "LOGS"}}
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:16]


def code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((config.ROOT / "src" / "voc").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------
# core: run everything on one dataset
# ---------------------------------------------------------------------------------------------------------------
def run_dataset(ds: data.Dataset, rows: np.ndarray, n_seeds: int, period_masks: dict[str, np.ndarray],
                label: str = "", log_every: int = 10) -> dict:
    """VoC over all seeds (both return conventions share the features) plus all baselines.

    Returns, per convention: the realised series y, the 'mean over seeds' statistics per grid cell and period,
    the per-seed Sharpe of the target cell, the ensemble (seed-averaged) forecasts, linear and simple positions.
    """
    n = len(rows)
    series = {"std": ds.R, "raw": ds.r}
    ys = {c: series[c][rows + 1] for c in config.CONVENTIONS}
    acc = {c: {p: {k: np.zeros((NP, NZ)) for k in STAT_KEYS} for p in period_masks} for c in config.CONVENTIONS}
    acc_sq = {c: {p: np.zeros((NP, NZ)) for p in period_masks} for c in config.CONVENTIONS}
    seed_sr = {c: {p: [] for p in period_masks} for c in config.CONVENTIONS}
    ens = {c: np.zeros((NP, NZ, n)) for c in config.CONVENTIONS}
    t0 = time.time()
    for s in range(n_seeds):
        f = model.voc_forecasts(ds, rows, s, targets=[series[c] for c in config.CONVENTIONS[1:]])
        for ci, c in enumerate(config.CONVENTIONS):
            fc = f[ci]
            ens[c] += fc
            flat = fc.reshape(-1, n)
            for p, mask in period_masks.items():
                st = metrics.batch_stats(flat[:, mask], ys[c][mask], ys[c][mask])
                for k in STAT_KEYS:
                    acc[c][p][k] += st[k].reshape(NP, NZ)
                sr = st["sharpe"].reshape(NP, NZ)
                acc_sq[c][p] += sr**2
                seed_sr[c][p].append(float(sr[KT, JT]))
        if (s + 1) % log_every == 0 or s + 1 == n_seeds:
            el = time.time() - t0
            log.info("%s seed %d/%d  %.0fs elapsed, eta %.0fs", label, s + 1, n_seeds, el, el / (s + 1) * (n_seeds - s - 1))
    out = {"rows": rows, "months": ds.target_months(rows), "n_seeds": n_seeds, "conv": {}}
    for c in config.CONVENTIONS:
        kmz = {p: {k: acc[c][p][k] / n_seeds for k in STAT_KEYS} for p in period_masks}
        for p in period_masks:
            kmz[p]["sharpe_seed_sd"] = np.sqrt(np.maximum(acc_sq[c][p] / n_seeds - kmz[p]["sharpe"] ** 2, 0))
        out["conv"][c] = {
            "y": ys[c], "kmz": kmz, "seed_sr": {p: np.array(v) for p, v in seed_sr[c].items()},
            "ens": ens[c] / n_seeds,
            "linear": model.linear_forecasts(ds, rows, target=series[c]),
            "simple": model.simple_forecasts(ds, rows, target=series[c]),
        }
    return out


def strategy_positions(cres: dict) -> dict[str, np.ndarray]:
    """All reported strategies of one convention (ensemble convention for VoC), full sample of the run."""
    n = len(cres["y"])
    pos = {"Buy-and-hold": np.ones(n)}
    for name in ("hist_mean",):
        pos[SIMPLE_LABELS[name]] = cres["simple"][name]
    for j, z in enumerate(config.LINEAR_Z):
        pos[f"Linear ridge, z={_zlabel(z)}"] = cres["linear"][j]
    for name in ("tsmom_sign", "mom12", "kmz_mom", "nagel_volmom"):
        pos[SIMPLE_LABELS[name]] = cres["simple"][name]
    for k, P in enumerate(config.P_GRID):
        for j, z in enumerate(config.Z_GRID):
            pos[voc_name(P, z)] = cres["ens"][k, j]
    return pos


def main_strategy_names() -> list[str]:
    names = ["Buy-and-hold", "Historical mean", "Linear ridge, z=0+", "Linear ridge, z=1e3"]
    names += [SIMPLE_LABELS[k] for k in ("tsmom_sign", "mom12", "kmz_mom", "nagel_volmom")]
    names += [voc_name(P, z) for P, z in MAIN_VOC]
    return names


def trial_names() -> list[str]:
    names = [f"Linear ridge, z={_zlabel(z)}" for z in config.LINEAR_Z]
    names += [SIMPLE_LABELS[k] for k in config.SIMPLE_RULES]
    names += [voc_name(P, z) for P in config.P_GRID for z in config.Z_GRID]
    return names  # 8 + 5 + 140 = 153


def is_forecast(name: str) -> bool:
    return name.startswith("VoC") or name.startswith("Linear") or name in {SIMPLE_LABELS[k] for k in FORECAST_STRATEGIES}


def evaluate(cres: dict, mask: np.ndarray, names: list[str] | None = None, bootstrap: bool = True) -> pd.DataFrame:
    """Full statistics (ensemble convention) for the main strategies on the months in `mask`."""
    y = cres["y"][mask]
    pos = {k: v[mask] for k, v in strategy_positions(cres).items()}
    trials = trial_names()
    srs = np.array([metrics.sharpe(pd.Series(pos[t] * y), 1) for t in trials])
    trial_var = float(np.var(srs, ddof=1))
    rows = []
    for name in names or main_strategy_names():
        st = metrics.full_stats(pos[name], y, y, trials=len(trials), trial_var=trial_var, bootstrap=bootstrap)
        if not is_forecast(name):
            st["r2"] = float("nan")
        if name == "Buy-and-hold":  # a regression of the market on itself: alpha is 0 by construction
            st.update({"alpha_ann": 0.0, "alpha_t": float("nan"), "alpha_t_nw": float("nan"), "ir": float("nan"), "beta": 1.0})
        st["strategy"] = name
        st["trials"] = len(trials)
        rows.append(st)
    df = pd.DataFrame(rows)
    return df[["strategy"] + [c for c in df.columns if c != "strategy"]]


def curve_frame(res: dict, period: str, mask: np.ndarray) -> pd.DataFrame:
    """Complexity curve: every (P, z) cell, 'mean over seeds' (paper) and ensemble statistics."""
    out = []
    for c in config.CONVENTIONS:
        cres = res["conv"][c]
        y = cres["y"][mask]
        kmz = cres["kmz"][period]
        ens = cres["ens"][:, :, mask]
        st = metrics.batch_stats(ens.reshape(-1, mask.sum()), y, y)
        for k, P in enumerate(config.P_GRID):
            for j, z in enumerate(config.Z_GRID):
                i = k * NZ + j
                row = {"conv": c, "P": P, "c": P / config.T, "z": z}
                row.update({f"{s}_paper": float(kmz[s][k, j]) for s in STAT_KEYS})
                row["sharpe_seed_sd"] = float(kmz["sharpe_seed_sd"][k, j])
                row.update({f"{s}_ens": float(st[s][i]) for s in ("sharpe", "r2", "alpha_t", "ir", "mean", "vol")})
                out.append(row)
    return pd.DataFrame(out)


def save_positions(res: dict, path, mask: np.ndarray | None = None) -> None:
    frames = []
    for c in config.CONVENTIONS:
        cres = res["conv"][c]
        pos = strategy_positions(cres)
        keep = main_strategy_names()
        df = pd.DataFrame({k: pos[k] for k in keep})
        df["y"] = cres["y"]
        df["month"] = res["months"].astype(str)
        df["conv"] = c
        frames.append(df if mask is None else df[mask])
    pd.concat(frames, ignore_index=True).to_parquet(path)


# ---------------------------------------------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------------------------------------------
def replicate(n_seeds: int | None = None) -> None:
    n_seeds = n_seeds or config.N_SEEDS_A
    t0 = time.time()
    ds = data.us_dataset(last_month=config.PAPER_LAST)  # the replication never sees a month after 2020-12
    rows = ds.forecast_rows(config.PAPER_FIRST, config.PAPER_LAST)
    months = ds.target_months(rows)
    log.info("replicate: %d OOS months %s..%s, %d seeds", len(rows), months[0], months[-1], n_seeds)
    masks = {"paper": np.ones(len(rows), bool)}
    res = run_dataset(ds, rows, n_seeds, masks, label="paper")
    R = config.RESULTS
    R.mkdir(parents=True, exist_ok=True)
    curve_frame(res, "paper", masks["paper"]).to_csv(R / "partA_paper_curve.csv", index=False)
    tabs = []
    for c in config.CONVENTIONS:
        t = evaluate(res["conv"][c], masks["paper"])
        t.insert(0, "conv", c)
        t.insert(1, "period", "paper")
        tabs.append(t)
    pd.concat(tabs).to_csv(R / "partA_paper_table.csv", index=False)
    save_positions(res, R / "partA_paper_positions.parquet")
    pd.DataFrame({c: res["conv"][c]["seed_sr"]["paper"] for c in config.CONVENTIONS}).to_csv(
        R / "partA_paper_seed_sharpe_target_cell.csv", index_label="seed")
    kmz = res["conv"]["std"]["kmz"]["paper"]
    check = {"target": "KMZ Table 1 Panel A, Nonlinear, T=12, P=12000, z=1e3: Sharpe 0.47 (mean over seeds)",
             "paper_sharpe": config.TARGET_SHARPE, "tolerance": config.TARGET_TOL,
             "ours_sharpe_mean_over_seeds": float(kmz["sharpe"][KT, JT]),
             "ours_sharpe_seed_sd": float(kmz["sharpe_seed_sd"][KT, JT]),
             "ours_alpha_t_mean_over_seeds": float(kmz["alpha_t"][KT, JT]),
             "ours_ir_mean_over_seeds": float(kmz["ir"][KT, JT]),
             "ours_r2_mean_over_seeds": float(kmz["r2"][KT, JT]),
             "raw_convention": {k: float(res["conv"]["raw"]["kmz"]["paper"][k][KT, JT]) for k in ("sharpe", "alpha_t", "ir")},
             "raw_convention_ridgeless": {k: float(res["conv"]["raw"]["kmz"]["paper"][k][KT, 0]) for k in ("sharpe", "alpha_t", "ir")},
             "nagel_2025_table1_raw_ridgeless": {"alpha_t": 2.417, "ir": 0.255},
             "n_seeds": n_seeds, "n_months": int(len(rows)), "first_target": str(months[0]), "last_target": str(months[-1]),
             "runtime_s": round(time.time() - t0, 1), "config_hash": config_hash(), "code_hash": code_hash()}
    check["within_tolerance"] = abs(check["ours_sharpe_mean_over_seeds"] - config.TARGET_SHARPE) <= config.TARGET_TOL
    (R / "replication_check.json").write_text(json.dumps(check, indent=2))
    log.info("replication: Sharpe %.3f (paper 0.47), within tolerance: %s; %.0fs", check["ours_sharpe_mean_over_seeds"],
             check["within_tolerance"], time.time() - t0)


def _read_log(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test(force: bool = False, n_seeds: int | None = None) -> None:
    """Score the post-sample months once (2021-01 .. last available target)."""
    n_seeds = n_seeds or config.N_SEEDS_A
    R = config.RESULTS
    if not (R / "partA_paper_table.csv").exists():
        raise SystemExit("run `replicate` first: the post-sample test comes after the replication")
    log_path = R / "test_log.jsonl"
    previous = [e for e in _read_log(log_path) if e.get("event") == "done"]
    if previous and not force:
        raise SystemExit("post-sample test already run (see results/test_log.jsonl); use --force to run again (logged)")
    t0 = time.time()
    stamp = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_hash": config_hash(),
             "code_hash": code_hash(), "forced_rerun": bool(previous), "n_seeds": n_seeds}
    with open(log_path, "a") as f:
        f.write(json.dumps({"event": "start", **stamp}) + "\n")
    ds = data.us_dataset()
    rows = ds.forecast_rows(config.POST_FIRST)
    months = ds.target_months(rows)
    log.info("test: %d post-sample months %s..%s", len(rows), months[0], months[-1])
    masks = {"post": np.ones(len(rows), bool), "post_jf": np.asarray(months >= pd.Period(config.POST_JF_FIRST, "M"))}
    res = run_dataset(ds, rows, n_seeds, masks, label="post")
    curves = [curve_frame(res, p, m).assign(period=p) for p, m in masks.items()]
    pd.concat(curves).to_csv(R / "partA_post_curve.csv", index=False)
    tabs = []
    for c in config.CONVENTIONS:
        for p, m in masks.items():
            t = evaluate(res["conv"][c], m)
            t.insert(0, "conv", c)
            t.insert(1, "period", p)
            tabs.append(t)
    tab = pd.concat(tabs)
    tab.to_csv(R / "partA_post_table.csv", index=False)
    save_positions(res, R / "partA_post_positions.parquet")
    head = tab[(tab.period == "post")].set_index(["conv", "strategy"])
    target = voc_name(config.TARGET_P, config.TARGET_Z)
    summary = {c: {"voc_sharpe": float(head.loc[(c, target), "sharpe"]), "bh_sharpe": float(head.loc[(c, "Buy-and-hold"), "sharpe"]),
                   "voc_alpha_t": float(head.loc[(c, target), "alpha_t"]), "voc_dsr": float(head.loc[(c, target), "dsr"]),
                   "voc_sharpe_paper_convention": float(res["conv"][c]["kmz"]["post"]["sharpe"][KT, JT])}
               for c in config.CONVENTIONS}
    entry = {"event": "done", **stamp, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "n_months": int(len(rows)), "first_target": str(months[0]), "last_target": str(months[-1]),
             "runtime_s": round(time.time() - t0, 1), "summary": summary}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("test done: %s", json.dumps(summary))


# ---------------------------------------------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------------------------------------------
def global_(n_seeds: int | None = None) -> None:
    n_seeds = n_seeds or config.N_SEEDS_B
    t0 = time.time()
    panels = data.load_jkp()
    R = config.RESULTS
    curves, tables, positions = [], [], []
    for i, (loc, panel) in enumerate(sorted(panels.items())):
        ds = data.jkp_dataset(panel, loc)
        rows = ds.forecast_rows()
        if len(rows) < config.JKP_MIN_OOS:
            log.info("skip %s: %d OOS months", loc, len(rows))
            continue
        mask = np.ones(len(rows), bool)
        res = run_dataset(ds, rows, n_seeds, {"full": mask}, label=loc, log_every=n_seeds)
        curves.append(curve_frame(res, "full", mask).assign(location=loc))
        for c in config.CONVENTIONS:
            t = evaluate(res["conv"][c], mask)
            t.insert(0, "conv", c)
            t.insert(0, "location", loc)
            tables.append(t)
            pos = strategy_positions(res["conv"][c])
            df = pd.DataFrame({k: pos[k] * res["conv"][c]["y"] for k in main_strategy_names()})  # strategy returns
            df["month"] = res["months"].astype(str)
            df["conv"] = c
            df["location"] = loc
            positions.append(df)
        log.info("[%d/%d] %s done (%d months, %s..%s), %.0fs total", i + 1, len(panels), loc, len(rows),
                 res["months"][0], res["months"][-1], time.time() - t0)
    pd.concat(curves).to_csv(R / "partB_curves.csv", index=False)
    pd.concat(tables).to_csv(R / "partB_country_table.csv", index=False)
    rets = pd.concat(positions, ignore_index=True)
    rets.to_parquet(R / "partB_strategy_returns.parquet")
    pooled(rets).to_csv(R / "partB_pooled_table.csv", index=False)
    meta = {"n_seeds": n_seeds, "runtime_s": round(time.time() - t0, 1), "config_hash": config_hash(), "code_hash": code_hash(),
            "locations": sorted(set(rets.location))}
    (R / "partB_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("global done in %.0fs", time.time() - t0)


def pooled(rets: pd.DataFrame) -> pd.DataFrame:
    """Equal-weighted portfolio across the non-US countries available each month, per strategy and convention."""
    out = []
    names = main_strategy_names()
    nonus = rets[rets.location != "usa"]
    trials = len(names)  # only the main strategies are pooled, so the trial family is these 13
    for c in config.CONVENTIONS:
        sub = nonus[nonus.conv == c]
        ew = sub.groupby("month")[names].mean().sort_index()
        n_ctry = sub.groupby("month").size()
        ew = ew[n_ctry.reindex(ew.index) >= 5]  # at least 5 countries in the pool
        mkt = ew["Buy-and-hold"].to_numpy()
        srs = np.array([metrics.sharpe(ew[k], 1) for k in names])
        tv = float(np.var(srs, ddof=1))
        for k in names:
            st = metrics.full_stats(ew[k].to_numpy(), np.ones(len(ew)), mkt, trials=trials, trial_var=tv)
            st["r2"] = float("nan")
            if k == "Buy-and-hold":
                st.update({"alpha_ann": 0.0, "alpha_t": float("nan"), "alpha_t_nw": float("nan"), "ir": float("nan"), "beta": 1.0})
            st.update({"conv": c, "strategy": k, "first": ew.index[0], "last": ew.index[-1]})
            out.append(st)
    df = pd.DataFrame(out)
    return df[["conv", "strategy"] + [x for x in df.columns if x not in ("conv", "strategy")]]
