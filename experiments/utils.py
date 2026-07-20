"""Output helpers and evaluation metrics."""
import os
import sys
import json
from functools import partial
from tqdm import tqdm as _tqdm

# Disable tqdm when running non-interactively or in GitHub Actions to avoid log clutter
tqdm = partial(_tqdm, disable=os.environ.get("GITHUB_ACTIONS") == "true" or not sys.stdout.isatty())

import numpy as np
from scipy.stats import spearmanr, pearsonr

from .config import CFG


def out_path(filename: str) -> str:
    os.makedirs(CFG.output_dir, exist_ok=True)
    return os.path.join(CFG.output_dir, filename)


def save_json(data, filename: str) -> str:
    p = out_path(filename)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    return p


def already_done(filename: str, force: bool = False) -> bool:
    return (not force) and os.path.exists(out_path(filename))


def _logistic_func(x, b1, b2, b3, b4, b5):
    lp = np.clip(b2 * (x - b3), -100, 100)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(lp))) + b4 * x + b5


def compute_metrics(preds, mos):
    from scipy.optimize import curve_fit

    p = np.array(preds, dtype=float)
    m = np.array(mos, dtype=float)
    valid = ~(np.isnan(p) | np.isnan(m))
    if valid.sum() < 2:
        return float("nan"), float("nan")
    pv, mv = p[valid], m[valid]

    srcc, _ = spearmanr(pv, mv)

    try:
        p0 = [np.max(mv), 10.0, np.mean(pv), 0.1, 0.1]
        popt, _ = curve_fit(_logistic_func, pv, mv, p0=p0, maxfev=10000)
        pv_mapped = _logistic_func(pv, *popt)
        plcc, _ = pearsonr(pv_mapped, mv)
    except Exception:
        plcc, _ = pearsonr(pv, mv)

    return float(srcc), float(plcc)
