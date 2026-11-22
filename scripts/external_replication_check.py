"""Read an independent public replication of KMZ and record how it standardises returns and what Sharpe it gets.

Repo: https://github.com/nkonts/replication-the-virtue-of-complexity-in-return-prediction (the only public code we found;
KMZ's own code is not public). Usage, from a separate folder (the repo is data, nothing in it is executed):

    git clone --depth 1 https://github.com/nkonts/replication-the-virtue-of-complexity-in-return-prediction.git <dir>
    uv run python scripts/external_replication_check.py <dir>

Writes results/external_replication_check.json, which the README generator reads.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/nkonts/replication-the-virtue-of-complexity-in-return-prediction"


def main(repo: Path) -> None:
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cells = json.loads((repo / "simulations.ipynb").read_text(encoding="utf-8"))["cells"]
    std_lines = [ln.strip() for c in cells if c["cell_type"] == "code" for ln in "".join(c["source"]).splitlines()
                 if "rolling(12).std()" in ln or "returns / returns_std" in ln]
    target_lines = sorted({ln.strip() for c in cells if c["cell_type"] == "code" for ln in "".join(c["source"]).splitlines()
                           if "y=returns.shift(-1)" in ln})
    assert std_lines and target_lines, "standardisation or target line not found; the repo changed"

    m = pd.read_parquet(repo / "data" / "metrics.parquet")
    cell = m[(m.gamma == 2) & (m.z.round(6) == 1000) & (m.c.round(6) == 1000)]
    out = {
        "repo": URL, "commit": commit,
        "standardisation_lines": std_lines, "target_lines": target_lines,
        "why_look_ahead": "returns.rolling(12).std() at month t+1 includes the month-t+1 return; the target is "
                          "returns.shift(-1), so the forecast made at t is scored on R(t+1) / std(R(t-10..t+1)).",
        "cell": {"gamma": 2, "z": 1000, "c": 1000, "P": int(cell.P.iloc[0]), "T": int(cell["T"].iloc[0])},
        "n_draws": int(cell.iteration.nunique()),
        "voc_sharpe_mean_over_draws": float(cell.SR.mean()),
        "voc_ir_mean_over_draws": float(cell.IR.mean()),
        "market_sharpe": float(cell["Market Sharpe Ratio"].mean()),
        "returns": "S&P index price returns (pct_change of the Welch-Goyal 'Index' column), not excess returns",
        "data_file": "PredictorData2021 - Monthly.csv (Welch-Goyal, to 2021-12)",
    }
    (ROOT / "results" / "external_replication_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
