"""Numbers for the research note (paper/note.tex), read from results/ only.

`python -m voc.note` writes paper/numbers.json and paper/numbers.tex (one LaTeX macro per number). The note never
types a number in; tests/test_note.py checks that numbers.json matches the results files and that every macro the
note uses is defined.
"""
import json

import pandas as pd

from voc import config

R = config.RESULTS
PAPER = config.ROOT / "paper"
TARGET = "VoC c=1000 (P=12000), z=1e3"
VM = "Nagel vol-timed momentum"


def _j(name: str) -> dict:
    return json.loads((R / name).read_text())


def compute() -> dict[str, str]:
    """Macro name -> formatted value (strings, exactly as printed in the note)."""
    f2 = lambda v: f"{v:.2f}".replace("-0.00", "0.00")  # noqa: E731
    f1 = lambda v: f"{v:.1f}".replace("-0.0", "0.0")  # noqa: E731
    n = {}
    pub = config.PAPER_TABLE1
    n.update(KmzSR=f2(pub["nonlinear_z1e3"]["SR"]), KmzT=f1(pub["nonlinear_z1e3"]["IR_t"]),
             KmzIR=f2(pub["nonlinear_z1e3"]["IR"]), KmzLinSR=f2(pub["linear_z1e3"]["SR"]),
             KmzLinT=f1(pub["linear_z1e3"]["IR_t"]), KmzLinIR=f2(pub["linear_z1e3"]["IR"]),
             KmzLinZero=f2(pub["linear_ridgeless"]["SR"]))

    chk = _j("replication_check.json")
    n.update(RepSR=f2(chk["ours_sharpe_mean_over_seeds"]), RepT=f1(chk["ours_alpha_t_mean_over_seeds"]),
             RepIR=f2(chk["ours_ir_mean_over_seeds"]), RepSeedSD=f"{chk['ours_sharpe_seed_sd']:.3f}",
             RepDraws=str(chk["n_seeds"]), RawT=f1(chk["raw_convention"]["alpha_t"]),
             RawIR=f2(chk["raw_convention"]["ir"]), RawZeroT=f2(chk["raw_convention_ridgeless"]["alpha_t"]),
             RawZeroIR=f2(chk["raw_convention_ridgeless"]["ir"]))

    d = _j("replication_diagnostics.json")
    la, lag = d["std_window_includes_forecast_month (look-ahead)"], d["std_trailing_t-11..t (planned)"]
    n.update(LaSR=f2(la["voc_z1e3"]["sharpe"]), LaT=f1(la["voc_z1e3"]["alpha_t"]), LaIR=f2(la["voc_z1e3"]["ir"]),
             LaBH=f2(la["bh_sharpe"]), LagBH=f2(lag["bh_sharpe"]), LaLinSR=f2(la["linear_z1e3"]["sharpe"]),
             LaLinT=f1(la["linear_z1e3"]["alpha_t"]), LaLinZero=f2(la["linear_ridgeless_sharpe"]),
             RmsSR=f2(d["rms_trailing_t-11..t"]["voc_z1e3"]["sharpe"]),
             ExtraLagSR=f2(d["std_trailing_t-12..t-1"]["voc_z1e3"]["sharpe"]),
             NagelRawT=f2(d["published"]["nagel_table1_rff_raw_ridgeless"]["alpha_t"]),
             NagelRawIR=f2(d["published"]["nagel_table1_rff_raw_ridgeless"]["ir"]))

    nk = _j("external_replication_check.json")
    n.update(NkSR=f2(nk["voc_sharpe_mean_over_draws"]), NkDraws=str(nk["n_draws"]), NkCommit=nk["commit"])

    ex = _j("external_data_diagnostic.json")
    e, o = ex["runs"]["ext data (its X, its R)"], ex["runs"]["our data (our X, our R)"]
    n.update(ExtSR=f2(e["voc_z1e3"]["sharpe"]), ExtT=f1(e["voc_z1e3"]["alpha_t"]), ExtIR=f2(e["voc_z1e3"]["ir"]),
             ExtLinSR=f2(e["linear_z1e3"]["sharpe"]), ExtLinT=f1(e["linear_z1e3"]["alpha_t"]),
             ExtLinIR=f2(e["linear_z1e3"]["ir"]), ExtLinZero=f2(e["linear_ridgeless_sharpe"]),
             OwnSR=f2(o["voc_z1e3"]["sharpe"]), OwnT=f1(o["voc_z1e3"]["alpha_t"]), OwnIR=f2(o["voc_z1e3"]["ir"]),
             OwnLinSR=f2(o["linear_z1e3"]["sharpe"]), OwnLinT=f1(o["linear_z1e3"]["alpha_t"]),
             OwnLinIR=f2(o["linear_z1e3"]["ir"]), OwnLinZero=f2(o["linear_ridgeless_sharpe"]),
             HybRetSR=f2(ex["runs"]["our X, its R"]["voc_z1e3"]["sharpe"]),
             HybPredSR=f2(ex["runs"]["its X, our R"]["voc_z1e3"]["sharpe"]),
             ExtDraws=str(ex["n_seeds"]), ExtCommit=ex["source"]["commit"],
             ExtSeedSD=f"{e['voc_z1e3']['sharpe_seed_sd']:.3f}",
             RrSR=f"{ex['published']['rrikrv_readme_rff_T12']['sharpe']:.3f}")
    c = ex["compare"]
    n.update(CorrInfl=f2(c["infl"]["corr"]), CorrLty=f2(c["lty"]["corr"]), CorrDp=f2(c["dp"]["corr"]),
             CorrEp=f2(c["ep"]["corr"]), CorrDfy=f2(c["dfy"]["corr"]), CorrR=f2(c["R (scaled return)"]["corr"]),
             CorrRet=f"{c['r (raw excess return)']['corr']:.3f}")

    b = _j("construction_b.json")
    keys = ["A", "One", "Two", "Three", "Four"]
    for k, (label, v) in zip(keys, b["decomposition_paper_sample"].items()):
        n[f"Dec{k}SR"] = f2(v["voc_z1e3_sharpe"])
        n[f"Dec{k}T"] = f1(v["voc_z1e3_alpha_t"])
        n[f"Dec{k}IR"] = f2(v["voc_z1e3_ir"])
        n[f"Dec{k}LinSR"] = f2(v["linear_z1e3_sharpe"])
        n[f"Dec{k}LinT"] = f1(v["linear_z1e3_alpha_t"])
        n[f"Dec{k}BH"] = f2(v["bh_sharpe"])
    srs = [v["voc_z1e3_sharpe"] for v in b["decomposition_paper_sample"].values()]
    steps = [y - x for x, y in zip(srs, srs[1:])]
    n.update(DecDraws="10", DecStepMin=f2(min(steps)), DecStepMax=f2(max(steps)))
    for con in ("A", "B"):
        for per, tag in (("paper", "In"), ("post", "Post")):
            v = b[con][per]
            n[f"{con}{tag}SR"] = f2(v["voc_z1e3_per_draw"]["sharpe"])
            n[f"{con}{tag}T"] = f1(v["voc_z1e3_per_draw"]["alpha_t"])
            n[f"{con}{tag}IR"] = f2(v["voc_z1e3_per_draw"]["ir"])
            n[f"{con}{tag}EnsSR"] = f2(v["voc_z1e3_ensemble"]["sharpe"])
            n[f"{con}{tag}EnsSE"] = f2(v["voc_z1e3_ensemble"]["sharpe_se"])
            n[f"{con}{tag}EnsT"] = f1(v["voc_z1e3_ensemble"]["alpha_t"])
            n[f"{con}{tag}Long"] = str(v["voc_z1e3_ensemble"]["months_long"])
            n[f"{con}{tag}BH"] = f2(v["bh_sharpe"])
            n[f"{con}{tag}BHSE"] = f2(v["bh_sharpe_se"])
            n[f"{con}{tag}LinSR"] = f2(v["linear_z1e3"]["sharpe"])
            n[f"{con}{tag}LinT"] = f1(v["linear_z1e3"]["alpha_t"])
            n[f"{con}{tag}N"] = str(v["n_months"])
    n["BDraws"] = str(b["B"]["n_draws"])
    n["PostFirst"], n["PostLast"] = b["B"]["post"]["first_target"], b["B"]["post"]["last_target"]

    post = pd.read_csv(R / "partA_post_table.csv")
    q = post[(post.conv == "std") & (post.period == "post")].set_index("strategy")
    n.update(OffN=str(int(q.loc[TARGET, "n"])), OffSR=f2(q.loc[TARGET, "sharpe"]), OffBH=f2(q.loc["Buy-and-hold", "sharpe"]),
             OffT=f1(q.loc[TARGET, "alpha_t"]), OffVmSR=f2(q.loc[VM, "sharpe"]))

    t = pd.read_csv(R / "partB_country_table.csv")
    t = t[(t.conv == "std") & (t.location != "usa")].set_index(["location", "strategy"])
    locs = sorted(set(t.index.get_level_values(0)))
    n.update(Ctry=str(len(locs)),
             CtryWins=str(sum(t.loc[(l, TARGET), "sharpe"] > t.loc[(l, "Buy-and-hold"), "sharpe"] for l in locs)),
             CtryVm=str(sum(t.loc[(l, VM), "sharpe"] > t.loc[(l, TARGET), "sharpe"] for l in locs)))
    pooled = pd.read_csv(R / "partB_pooled_table.csv").set_index(["conv", "strategy"])
    n["PoolT"] = f1(pooled.loc[("std", TARGET), "alpha_t"])
    return n


def write() -> None:
    n = compute()
    PAPER.mkdir(exist_ok=True)
    (PAPER / "numbers.json").write_text(json.dumps(n, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["% generated by `python -m voc.note` from results/; do not edit"]
    lines += [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in sorted(n.items())]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write()
