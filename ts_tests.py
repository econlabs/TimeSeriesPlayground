"""
Classical hypothesis and diagnostic tests for univariate time series.

Each ``run_*`` function returns either ``None`` (not enough data / failure) or a
small dictionary of labelled outputs suitable for a single row in a pandas
DataFrame. ``series_test_table`` and ``residual_test_table`` bundle those rows
for the Streamlit "Statistical tests" tab.

Libraries:
    - ``scipy.stats`` for normality and K–S.
    - ``statsmodels.tsa.stattools`` for ADF, KPSS, Zivot–Andrews, BDS, break-variance.
    - ``statsmodels.stats.diagnostic`` for Ljung–Box and ARCH LM.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, bds, breakvar_heteroskedasticity_test, kpss, zivot_andrews


def _finite(x: np.ndarray) -> np.ndarray:
    """
    Drop NaN/inf from a 1D array and return a float64 view.

    Args:
        x: Input array (any numeric dtype).

    Returns:
        1D array containing only finite values.
    """
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def run_adf(x: np.ndarray) -> dict[str, Any] | None:
    """
    Augmented Dickey–Fuller unit-root test (null: non-stationary / unit root).

    Args:
        x: Time series (transformed as desired).

    Returns:
        Dict with test name, H0 text, statistic, p-value, lags, n, and critical
        values, or ``None`` if ``len(x) < 20``.
    """
    x = _finite(x)
    if len(x) < 20:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = adfuller(x, autolag="AIC")
    cv = r[4]
    return {
        "Test": "Augmented Dickey–Fuller",
        "H0": "Unit root (non-stationary)",
        "Statistic": float(r[0]),
        "p-value": float(r[1]),
        "Lags used": int(r[2]),
        "n": int(r[3]),
        "Crit 1%": float(cv["1%"]),
        "Crit 5%": float(cv["5%"]),
        "Crit 10%": float(cv["10%"]),
    }


def run_kpss(x: np.ndarray) -> dict[str, Any] | None:
    """
    KPSS stationarity test (null: series is stationary around a constant).

    Note:
        ADF and KPSS have **opposite** null hypotheses—read them together.

    Args:
        x: Time series.

    Returns:
        Result dict or ``None`` if too few observations.
    """
    x = _finite(x)
    if len(x) < 20:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lag, crit = kpss(x, regression="c", nlags="auto")
    return {
        "Test": "KPSS",
        "H0": "Stationarity (trend-stationary around constant)",
        "Statistic": float(stat),
        "p-value": float(p),
        "Lags": int(lag),
        "Crit 10%": float(crit["10%"]),
        "Crit 5%": float(crit["5%"]),
        "Crit 2.5%": float(crit["2.5%"]),
        "Crit 1%": float(crit["1%"]),
    }


def run_zivot_andrews(x: np.ndarray) -> dict[str, Any] | None:
    """
    Zivot–Andrews unit-root test allowing one structural break (constant only).

    Args:
        x: Time series.

    Returns:
        Dict with statistic, p-value, break index, etc., or ``None`` if too
        short or the routine raises.
    """
    x = _finite(x)
    if len(x) < 40:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zastat, pvalue, cvdict, baselag, bpidx = zivot_andrews(x, regression="c", autolag="AIC")
    except Exception:
        return None
    return {
        "Test": "Zivot–Andrews",
        "H0": "Unit root with a single break",
        "Statistic": float(zastat),
        "p-value": float(pvalue),
        "Lags": int(baselag),
        "Break index": int(bpidx),
        "Crit 1%": float(cvdict["1%"]),
        "Crit 5%": float(cvdict["5%"]),
        "Crit 10%": float(cvdict["10%"]),
    }


def run_shapiro(x: np.ndarray) -> dict[str, Any] | None:
    """
    Shapiro–Wilk normality test.

    SciPy recommends limiting length; we use the last 5,000 points if longer.

    Args:
        x: Data (e.g. transformed series).

    Returns:
        Dict with statistic and p-value, or ``None`` if ``len(x) < 3``.
    """
    x = _finite(x)
    if len(x) < 3:
        return None
    note = ""
    if len(x) > 5000:
        x = x[-5000:]
        note = "Last 5,000 observations (Shapiro–Wilk limit)."
    stat, p = stats.shapiro(x)
    out = {
        "Test": "Shapiro–Wilk",
        "H0": "Normality",
        "Statistic": float(stat),
        "p-value": float(p),
        "n used": len(x),
    }
    if note:
        out["Note"] = note
    return out


def run_anderson_darling(x: np.ndarray) -> dict[str, Any] | None:
    """
    Anderson–Darling test for normality (with interpolated p-value).

    Args:
        x: Sample values.

    Returns:
        Dict with statistic and p-value, or ``None`` if ``len(x) < 8``.
    """
    x = _finite(x)
    if len(x) < 8:
        return None
    r = stats.anderson(x, dist="norm", method="interpolate")
    return {
        "Test": "Anderson–Darling",
        "H0": "Sample from normal distribution",
        "Statistic": float(r.statistic),
        "p-value": float(r.pvalue),
        "n": len(x),
    }


def run_ks_standard_normal(x: np.ndarray) -> dict[str, Any] | None:
    """
    Kolmogorov–Smirnov test: standardized data vs standard normal.

    We z-score ``x`` first, then test against ``N(0,1)``. This is a simple
    visual companion to the Q–Q plot, not a formal test with estimated μ,σ on
    the same sample (that would need Lilliefors).

    Args:
        x: Sample.

    Returns:
        Dict with K–S statistic and p-value, or ``None`` if too few points.
    """
    x = _finite(x)
    if len(x) < 5:
        return None
    z = (x - np.mean(x)) / np.std(x, ddof=1)
    res = stats.kstest(z, "norm")
    return {
        "Test": "Kolmogorov–Smirnov (standardized)",
        "H0": "Standardized series ~ N(0,1)",
        "Statistic": float(res.statistic),
        "p-value": float(res.pvalue),
        "n": len(x),
    }


def run_ljung_box(x: np.ndarray, lags: int = 10, model_df: int = 0) -> dict[str, Any] | None:
    """
    Ljung–Box test for autocorrelation up to ``lags``.

    When testing ARIMA residuals, pass ``model_df=p+q`` so degrees of freedom
    are adjusted.

    Args:
        x: Series or residuals.
        lags: Maximum lag included in the test.
        model_df: Number of estimated AR+MA parameters to subtract (0 for raw series).

    Returns:
        One-row-style dict or ``None`` if the series is too short.
    """
    x = _finite(x)
    if len(x) <= lags + 5:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = acorr_ljungbox(x, lags=[lags], model_df=model_df, return_df=True)
    row = df.iloc[0]
    return {
        "Test": f"Ljung–Box (lag {lags})",
        "H0": "No serial correlation up to lag",
        "Statistic": float(row["lb_stat"]),
        "p-value": float(row["lb_pvalue"]),
        "model_df": model_df,
        "n": len(x),
    }


def run_bds(x: np.ndarray, max_n: int = 500, max_dim: int = 2) -> dict[str, Any] | None:
    """
    BDS test for i.i.d. (detects general dependence including non-linear).

    The implementation is heavy for long series; we keep only the last ``max_n``
    points when needed.

    Args:
        x: Series or residuals.
        max_n: If ``len(x)`` exceeds this, use the tail slice only.
        max_dim: Embedding dimension passed to statsmodels.

    Returns:
        Dict with BDS statistic and p-value, or ``None`` if ``len(x) < 50``.
    """
    x = _finite(x)
    if len(x) < 50:
        return None
    note = ""
    if len(x) > max_n:
        x = x[-max_n:]
        note = f"Last {max_n} observations (BDS is costly for long series)."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bstat, pval = bds(x, max_dim=max_dim)
    out = {
        "Test": "BDS",
        "H0": "i.i.d. (no general dependence)",
        "Statistic": float(bstat),
        "p-value": float(pval),
        "max_dim": max_dim,
        "n used": len(x),
    }
    if note:
        out["Note"] = note
    return out


def run_arch_lm(resid: np.ndarray, nlags: int = 5) -> dict[str, Any] | None:
    """
    Engle's ARCH LM test on residuals (null: no ARCH effects up to ``nlags``).

    Args:
        resid: Residual series (e.g. from ARIMA ``.resid``).
        nlags: Number of lags in the auxiliary regression.

    Returns:
        Dict with LM and F versions of the test, or ``None`` if too few points.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < nlags + 15:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm, lmpval, fval, fpval = het_arch(r, nlags=nlags, store=False)
    return {
        "Test": "ARCH LM (Engle)",
        "H0": "No ARCH up to selected lags",
        "LM statistic": float(lm),
        "LM p-value": float(lmpval),
        "F statistic": float(fval),
        "F p-value": float(fpval),
        "nlags": nlags,
        "n": len(r),
    }


def run_breakvar(resid: np.ndarray) -> dict[str, Any] | None:
    """
    Break-variance test (Goldfeld–Quandt style on early vs late subsamples).

    Args:
        resid: Residual series.

    Returns:
        Dict with statistic and p-value, or ``None`` if too few observations.
    """
    r = np.asarray(resid, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 40:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval = breakvar_heteroskedasticity_test(r)
    return {
        "Test": "Break-variance (Goldfeld–Quandt type)",
        "H0": "Homoskedasticity (equal variance early vs late sample)",
        "Statistic": float(stat),
        "p-value": float(pval),
        "n": len(r),
    }


def series_test_table(x: np.ndarray, ljung_lag: int = 10) -> pd.DataFrame:
    """
    Run the standard battery on the **observed** (transformed) series.

    Order: ADF, KPSS, Zivot–Andrews, Shapiro, Anderson–Darling, K–S,
    Ljung–Box, BDS.

    Args:
        x: 1D observations.
        ljung_lag: Lag used in Ljung–Box.

    Returns:
        DataFrame with one row per successful test (possibly empty).
    """
    rows: list[dict[str, Any]] = []
    for fn in (run_adf, run_kpss, run_zivot_andrews, run_shapiro, run_anderson_darling, run_ks_standard_normal):
        r = fn(x)
        if r:
            rows.append(r)
    lb = run_ljung_box(x, lags=ljung_lag, model_df=0)
    if lb:
        rows.append(lb)
    bd = run_bds(x)
    if bd:
        rows.append(bd)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def residual_test_table(resid: np.ndarray, order: tuple[int, int, int], ljung_lag: int = 10) -> pd.DataFrame:
    """
    Run diagnostics on **ARIMA residuals**.

    Ljung–Box uses ``model_df = p + q`` from ``order``.

    Args:
        resid: Residual array (may contain NaNs at start—callers often pass raw ``.resid``).
        order: Fitted ``(p, d, q)``.
        ljung_lag: Ljung–Box maximum lag.

    Returns:
        DataFrame with Ljung–Box, ARCH LM, break-variance, and BDS rows when available.
    """
    p, d, q = order
    model_df = p + q
    rows: list[dict[str, Any]] = []
    lb = run_ljung_box(resid, lags=ljung_lag, model_df=model_df)
    if lb:
        rows.append(lb)
    arch = run_arch_lm(resid)
    if arch:
        rows.append(arch)
    bv = run_breakvar(resid)
    if bv:
        rows.append(bv)
    bd_res = run_bds(resid, max_n=500, max_dim=2)
    if bd_res:
        row = dict(bd_res)
        row["Test"] = "BDS (residuals)"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
