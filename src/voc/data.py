"""Build stage: turn the raw downloads into clean monthly panels, and turn a panel into a model dataset.

A model dataset holds, for consecutive months s:
  X[s]  predictors of month s divided by their expanding-window std (data up to s only, >= 36 observations),
  R[s]  excess return of month s divided by the std of the 12 returns before s (data up to s-1 only),
  r[s]  raw excess return of month s.
Both scalings only use past data, so a forecast made at month t never sees anything after t.
"""

import io
import logging
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from voc import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------------------------
# raw readers
# ---------------------------------------------------------------------------------------------------------------
def read_goyal(path=config.GOYAL_FILE) -> pd.DataFrame:
    """Welch-Goyal monthly sheet -> the 14 KMZ predictors built with the Welch-Goyal (2008) definitions."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl header/footer warning
        g = pd.read_excel(path, sheet_name="Monthly")
    g.index = pd.PeriodIndex([pd.Period(f"{int(v) // 100}-{int(v) % 100:02d}", "M") for v in g["yyyymm"]], name="month")
    p = g["price"].astype(float)
    out = pd.DataFrame(index=g.index)
    out["dp"] = np.log(g["d12"]) - np.log(p)
    out["dy"] = np.log(g["d12"]) - np.log(p.shift(1))
    out["ep"] = np.log(g["e12"]) - np.log(p)
    out["de"] = np.log(g["d12"]) - np.log(g["e12"])
    out["svar"] = g["svar"]
    out["bm"] = g["b/m"]
    out["ntis"] = g["ntis"]
    out["tbl"] = g["tbl"]
    out["lty"] = g["lty"]
    out["ltr"] = g["ltr"]
    out["tms"] = g["lty"] - g["tbl"]
    out["dfy"] = g["BAA"] - g["AAA"]
    out["dfr"] = g["corpr"] - g["ltr"]
    out["infl"] = g["infl"].shift(config.INFL_EXTRA_LAG)
    return out.astype(float)


def read_french(path=config.FRENCH_FILE) -> pd.DataFrame:
    """Monthly Mkt-RF and RF (decimals) from the Fama-French research factors CSV."""
    with zipfile.ZipFile(path) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    started = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(",Mkt-RF"):
            started = True
            continue
        if started:
            if not s or not s[:6].isdigit() or len(s.split(",")[0].strip()) != 6:
                break
            parts = [x.strip() for x in s.split(",")]
            rows.append((parts[0], float(parts[1]) / 100, float(parts[4]) / 100))
    df = pd.DataFrame(rows, columns=["yyyymm", "mkt_rf", "rf"])
    df.index = pd.PeriodIndex([pd.Period(f"{v[:4]}-{v[4:]}", "M") for v in df["yyyymm"]], name="month")
    return df[["mkt_rf", "rf"]]


def _read_zip_csv(path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))


def read_jkp() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = _read_zip_csv(config.JKP_MKT_FILE)
    t = _read_zip_csv(config.JKP_THEMES_FILE)
    m["month"] = pd.PeriodIndex(pd.to_datetime(m["date"]), freq="M")
    t["month"] = pd.PeriodIndex(pd.to_datetime(t["date"]), freq="M")
    return m, t


# ---------------------------------------------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------------------------------------------
def build_us() -> pd.DataFrame:
    """US panel: 15 raw predictors (to the last Goyal month) and the excess return r (to the last French month).

    The panel starts in 1926-07, the first CRSP month; earlier Goyal data are not used.
    """
    g = read_goyal()
    f = read_french()
    months = pd.period_range(f.index.min(), f.index.max(), freq="M")
    panel = pd.DataFrame(index=months)
    panel.index.name = "month"
    panel = panel.join(g, how="left")
    panel["lag_mkt"] = f["mkt_rf"]
    panel["r"] = f["mkt_rf"]
    last_goyal = g.dropna(how="all").index.max()
    panel.loc[panel.index > last_goyal, list(config.US_PREDICTORS)] = np.nan
    return panel[list(config.US_PREDICTORS) + ["r"]]


def build_jkp() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long JKP panel of the selected countries and the selection table (PLAN.md section 9)."""
    m, t = read_jkp()
    themes = t.pivot_table(index=["location", "month"], columns="name", values="ret")
    mk = m.set_index(["location", "month"])[["ret", "n_stocks"]].rename(columns={"ret": "r"})
    df = mk.join(themes, how="left")
    sel_rows, panels = [], []
    for loc, g in df.groupby(level=0):
        g = g.droplevel(0).sort_index()
        ok = g[list(config.JKP_THEMES)].notna().all(axis=1) & g["r"].notna() & (g["n_stocks"] >= config.JKP_MIN_STOCKS)
        months = g.index[ok]
        best = (0, None, None)
        if len(months):
            run_start = months[0]
            prev = months[0]
            length = 1
            for mo in months[1:]:
                if (mo - prev).n == 1:
                    length += 1
                else:
                    if length > best[0]:
                        best = (length, run_start, prev)
                    run_start, length = mo, 1
                prev = mo
            if length > best[0]:
                best = (length, run_start, prev)
        length, first, last = best
        row = {"location": loc, "run_months": length, "first": str(first) if first else None,
               "last": str(last) if last else None, "median_n_stocks": float(g["n_stocks"].median())}
        if length:
            sub = g.loc[first:last].copy()
            sub["lag_mkt"] = sub["r"]
            sub["mkt_vol12"] = sub["r"].rolling(config.RET_VOL_WINDOW).std(ddof=1)
            ds = make_dataset(sub[list(config.JKP_PREDICTORS)], sub["r"], name=loc)
            n_oos = len(ds.forecast_rows())
            row["n_oos"] = n_oos
            if n_oos >= config.JKP_MIN_OOS:
                sub = sub[list(config.JKP_PREDICTORS) + ["r", "n_stocks"]]
                sub.insert(0, "location", loc)
                panels.append(sub.reset_index())
        else:
            row["n_oos"] = 0
        row["included"] = row["n_oos"] >= config.JKP_MIN_OOS
        sel_rows.append(row)
    sel = pd.DataFrame(sel_rows).sort_values(["included", "n_oos"], ascending=[False, False])
    return pd.concat(panels, ignore_index=True), sel


def build_all() -> None:
    config.BUILT.mkdir(parents=True, exist_ok=True)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    us = build_us()
    us_out = us.copy()
    us_out.index = us_out.index.astype(str)
    us_out.to_parquet(config.BUILT / "us_panel.parquet")
    valid = us[list(config.US_PREDICTORS)].dropna()
    log.info("US panel %s..%s, predictors complete %s..%s, returns to %s", us.index.min(), us.index.max(),
             valid.index.min(), valid.index.max(), us["r"].dropna().index.max())
    jkp, sel = build_jkp()
    jkp["month"] = jkp["month"].astype(str)
    jkp.to_parquet(config.BUILT / "jkp_panel.parquet")
    sel.to_csv(config.RESULTS / "partB_country_selection.csv", index=False)
    log.info("JKP: %d locations included (%d non-US) of %d", sel["included"].sum(),
             int(sel["included"].sum()) - int(sel.loc[sel.location == "usa", "included"].sum()), len(sel))


def load_us() -> pd.DataFrame:
    df = pd.read_parquet(config.BUILT / "us_panel.parquet")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


def load_jkp() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(config.BUILT / "jkp_panel.parquet")
    out = {}
    for loc, g in df.groupby("location"):
        g = g.set_index(pd.PeriodIndex(g["month"], freq="M")).drop(columns=["month", "location"])
        out[loc] = g
    return out


# ---------------------------------------------------------------------------------------------------------------
# model dataset
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Dataset:
    name: str
    months: pd.PeriodIndex
    X: np.ndarray  # (N, K) scaled predictors
    R: np.ndarray  # (N,) standardised excess return
    r: np.ndarray  # (N,) raw excess return

    def forecast_rows(self, first_target=None, last_target=None, T: int = config.T) -> np.ndarray:
        """Rows m (forecast made at the end of month m for month m+1) with a full window and a known outcome."""
        N = len(self.months)
        xok = np.isfinite(self.X).all(axis=1)
        rok = np.isfinite(self.R)
        rows = []
        for m in range(T, N - 1):
            if not (xok[m - T:m + 1].all() and rok[m - T + 1:m + 2].all() and np.isfinite(self.r[m - T + 1:m + 1]).all()):
                continue
            tgt = self.months[m + 1]
            if first_target is not None and tgt < pd.Period(first_target, "M"):
                continue
            if last_target is not None and tgt > pd.Period(last_target, "M"):
                continue
            rows.append(m)
        return np.asarray(rows, dtype=int)

    def target_months(self, rows) -> pd.PeriodIndex:
        return self.months[np.asarray(rows) + 1]


def scale_predictors(pred: pd.DataFrame, min_obs: int = config.PRED_MIN_OBS) -> pd.DataFrame:
    """x_s / std(x_first..x_s): expanding window, includes month s, no demeaning."""
    sd = pred.expanding(min_periods=min_obs).std(ddof=1)
    return pred / sd


def scale_returns(r: pd.Series, window: int = config.RET_VOL_WINDOW) -> pd.Series:
    """R_s = r_s / std(r_{s-window} .. r_{s-1})."""
    return r / r.rolling(window).std(ddof=1).shift(1)


def make_dataset(pred: pd.DataFrame, r: pd.Series, name: str = "") -> Dataset:
    """pred and r on the same consecutive monthly index."""
    idx = pred.index
    assert (np.diff(idx.asi8) == 1).all(), "months must be consecutive"
    X = scale_predictors(pred).to_numpy(dtype=float)
    R = scale_returns(r.reindex(idx)).to_numpy(dtype=float)
    return Dataset(name=name, months=idx, X=X, R=R, r=r.reindex(idx).to_numpy(dtype=float))


def us_dataset(panel: pd.DataFrame | None = None, last_month: str | None = None) -> Dataset:
    """US dataset; `last_month` cuts the raw data before any scaling (the replication never sees later months)."""
    p = load_us() if panel is None else panel
    if last_month is not None:
        p = p.loc[: pd.Period(last_month, "M")]
    return make_dataset(p[list(config.US_PREDICTORS)], p["r"], name="usa_goyal")


def jkp_dataset(panel: pd.DataFrame, name: str) -> Dataset:
    return make_dataset(panel[list(config.JKP_PREDICTORS)], panel["r"], name=name)
