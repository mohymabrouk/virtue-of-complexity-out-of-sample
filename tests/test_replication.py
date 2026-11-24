"""Test 6: the replication target (KMZ Table 1, Panel A, Nonlinear, T=12, P=12,000, z=10^3: Sharpe 0.47).

Tolerance fixed in PLAN.md before any result: +/-0.10 on the Sharpe computed the paper's way (per-seed Sharpe,
averaged over seeds). Our number is about 0.30, so the test is marked xfail; README section "Replication"
explains the gap. The tolerance is not loosened.

A second check, written after the gap was found and stated as such in PLAN.md: in the raw-return convention our
ridgeless P=12,000 strategy reproduces the alpha t-stat and IR that Nagel (2025, Table I) obtained with KMZ's own
code (2.417 and 0.255), within +/-0.5 (t) and +/-0.06 (IR).
"""

import json

import pytest

from voc import config

CHECK = config.RESULTS / "replication_check.json"


def _check():
    return json.loads(CHECK.read_text())


@pytest.mark.skipif(not CHECK.exists(), reason="run `python -m voc.cli replicate` first")
@pytest.mark.xfail(strict=True, reason="Sharpe ~0.31 vs KMZ 0.47 with the pre-registered data construction; "
                   "the gap comes from four data choices (README.md, Replication, item 6)")
def test_replication_target_sharpe():
    c = _check()
    assert abs(c["ours_sharpe_mean_over_seeds"] - config.TARGET_SHARPE) <= config.TARGET_TOL


@pytest.mark.skipif(not CHECK.exists(), reason="run `python -m voc.cli replicate` first")
def test_raw_convention_matches_kmz_code_via_nagel():
    c = _check()
    assert abs(c["raw_convention_ridgeless"]["alpha_t"] - 2.417) <= 0.5
    assert abs(c["raw_convention_ridgeless"]["ir"] - 0.255) <= 0.06
