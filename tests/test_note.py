"""The research note types no number in: paper/numbers.json must equal what voc.note computes from results/, and
every number macro the note uses must be defined."""

import json
import re

import pytest

from voc import config, note

NUMBERS = config.ROOT / "paper" / "numbers.json"
NEEDED = ["replication_check.json", "replication_diagnostics.json", "external_replication_check.json",
          "external_data_diagnostic.json", "construction_b.json", "partA_post_table.csv",
          "partB_country_table.csv", "partB_pooled_table.csv"]
missing = [f for f in NEEDED if not (config.RESULTS / f).exists()]
pytestmark = pytest.mark.skipif(bool(missing) or not NUMBERS.exists(), reason=f"results missing: {missing}")


def test_numbers_match_results():
    assert json.loads(NUMBERS.read_text(encoding="utf-8")) == note.compute()


def test_every_macro_defined():
    tex = (config.ROOT / "paper" / "note.tex").read_text(encoding="utf-8")
    defined = set(json.loads(NUMBERS.read_text(encoding="utf-8")))
    used = set(re.findall(r"\\n([A-Z][A-Za-z]*)", tex))
    assert used, "the note uses no number macros"
    assert used <= defined, sorted(used - defined)


def test_construction_b_reproduces_target():
    """Stated after the fact (PLAN.md 16): construction B is within the pre-registered tolerance of KMZ's 0.47."""
    b = json.loads((config.RESULTS / "construction_b.json").read_text())
    assert b["builder_reproduces_A"]
    sr = b["B"]["paper"]["voc_z1e3_per_draw"]["sharpe"]
    assert abs(sr - config.TARGET_SHARPE) <= config.TARGET_TOL
