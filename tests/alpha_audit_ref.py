"""Load Alpha Audit's `sharpe` and `deflated_sharpe` from their source text, without importing the package.

The functions are pulled out of C:/dev/quant-agent-lab/backend/src/qal/lab/metrics.py with `ast` and executed in a
namespace that holds only what they use. Returns None when that file is not on this machine.
"""

import ast
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ALPHA_AUDIT_METRICS = Path("C:/dev/quant-agent-lab/backend/src/qal/lab/metrics.py")


def load():
    if not ALPHA_AUDIT_METRICS.exists():
        return None
    tree = ast.parse(ALPHA_AUDIT_METRICS.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("sharpe", "deflated_sharpe")]
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("EULER_GAMMA", "_NORMAL") for t in n.targets)]
    module = ast.Module(body=consts + wanted, type_ignores=[])
    ns = {"math": math, "np": np, "pd": pd, "NormalDist": NormalDist}
    exec(compile(module, str(ALPHA_AUDIT_METRICS), "exec"), ns)
    return ns["sharpe"], ns["deflated_sharpe"]
