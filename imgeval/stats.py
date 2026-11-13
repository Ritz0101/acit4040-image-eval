"""Small-sample statistics for the human study.

With 5 raters, trials are not independent (each rater answers many). So every headline number is reported two ways:
  pooled over trials (binomial test, Wilson CI), and with a cluster bootstrap over raters (the honest CI).
The proper model for a larger study is a mixed-effects logistic regression with a random intercept per rater.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest, spearmanr


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def rate(successes: list[bool], raters: list[str], p0: float = 0.5, alternative: str = "greater",
         n_boot: int = 5000, seed: int = 0) -> dict:
    """Win/fooling rate with a binomial test vs p0, Wilson CI, and a rater-level cluster bootstrap CI."""
    x = np.asarray(successes, dtype=bool)
    k, n = int(x.sum()), len(x)
    if n == 0:
        return {"n": 0}
    lo, hi = wilson(k, n)
    res = {"n": n, "k": k, "rate": k / n, "wilson_lo": lo, "wilson_hi": hi,
           "p_binom": binomtest(k, n, p0, alternative=alternative).pvalue}
    r = np.asarray(raters)
    ids = np.unique(r)
    per = {str(i): float(x[r == i].mean()) for i in ids}
    res["per_rater"] = per
    res["raters_above_p0"] = sum(v > p0 for v in per.values())
    if len(ids) >= 2:
        rng = np.random.default_rng(seed)
        groups = [x[r == i] for i in ids]
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(ids), len(ids))
            boots.append(np.concatenate([groups[j] for j in pick]).mean())
        res["boot_lo"], res["boot_hi"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return res


def mean_ci(values: list[float], raters: list[str] | None = None, n_boot: int = 5000, seed: int = 0) -> dict:
    v = np.asarray(values, dtype=float)
    v_ok = ~np.isnan(v)
    v = v[v_ok]
    if len(v) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    if raters is not None:
        r = np.asarray(raters)[v_ok]
        ids = np.unique(r)
        groups = [v[r == i] for i in ids]
        boots = [np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(n_boot)]
    else:
        boots = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "lo": float(lo), "hi": float(hi)}


def agreement(auto: list[float], human: list[float]) -> dict:
    """Spearman rank correlation between an automatic metric and the human mean per image."""
    a, h = np.asarray(auto, float), np.asarray(human, float)
    ok = ~(np.isnan(a) | np.isnan(h))
    if ok.sum() < 4:
        return {"n": int(ok.sum()), "rho": None, "p": None}
    r = spearmanr(a[ok], h[ok])
    return {"n": int(ok.sum()), "rho": float(r.statistic), "p": float(r.pvalue)}
