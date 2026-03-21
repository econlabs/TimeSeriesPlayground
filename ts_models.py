"""
ARIMA model search and refitting using statsmodels.

This module implements a **small grid search**: for each candidate order
``(p, d, q)`` it fits an ``ARIMA`` and records AIC/BIC. The best order is the
lowest AIC. A separate helper refits the chosen model with a pandas
``DatetimeIndex`` so plots align dates correctly.

Dependencies:
    ``statsmodels`` (``ARIMA`` in ``statsmodels.tsa.arima.model``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError as e:  # pragma: no cover
    raise ImportError("Install statsmodels: uv pip install statsmodels") from e


@dataclass(frozen=True)
class ArimaGridResult:
    """
    Summary of one successful ARIMA fit during the grid search.

    Attributes:
        order: Tuple ``(p, d, q)`` as in textbook ARIMA notation.
        aic: Akaike information criterion (lower is better for comparison).
        bic: Bayesian information criterion.
        llf: Log-likelihood at the optimum.
        nobs: Effective number of observations used in estimation.
        converged: Whether the optimizer reported convergence (when available).
    """

    order: tuple[int, int, int]
    aic: float
    bic: float
    llf: float
    nobs: float
    converged: bool


def _order_grid(model_type: str, max_p: int, max_q: int, max_d: int) -> Iterator[tuple[int, int, int]]:
    """
    Yield every ``(p, d, q)`` tuple to try for a given model family.

    The four families restrict which orders appear (e.g. AR only uses ``(p,0,0)``).

    Args:
        model_type: One of ``"AR"``, ``"MA"``, ``"ARMA"``, ``"ARIMA"``.
        max_p: Maximum autoregressive order to try (inclusive).
        max_q: Maximum moving-average order to try (inclusive).
        max_d: Maximum differencing order (only used when model_type is ``"ARIMA"``).

    Yields:
        ``(p, d, q)`` integer triples.

    Raises:
        ValueError: If ``model_type`` is not recognized.
    """
    if model_type == "AR":
        for p in range(1, max_p + 1):
            yield (p, 0, 0)
    elif model_type == "MA":
        for q in range(1, max_q + 1):
            yield (0, 0, q)
    elif model_type == "ARMA":
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                yield (p, 0, q)
    elif model_type == "ARIMA":
        for d in range(0, max_d + 1):
            for p in range(0, max_p + 1):
                for q in range(0, max_q + 1):
                    # Skip degenerate (0,0,0): not a meaningful model.
                    if p == 0 and q == 0 and d == 0:
                        continue
                    yield (p, d, q)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _trend_for_order(d: int) -> str:
    """
    Choose statsmodels ``trend`` string from the differencing order ``d``.

    When ``d == 0`` we include a constant (``"c"``). When ``d > 0`` the default
    is no separate constant in the same way (``"n"``)—this matches common practice
    for integrated models in this app.

    Args:
        d: Differencing order ``d`` from ``(p,d,q)``.

    Returns:
        ``"c"`` if ``d == 0``, else ``"n"``.
    """
    return "n" if d > 0 else "c"


def try_fit_arima(
    y: np.ndarray,
    order: tuple[int, int, int],
    *,
    trend: str | None = None,
) -> ArimaGridResult | None:
    """
    Fit a single ARIMA specification; return ``None`` if estimation fails.

    Failures (singular matrix, non-convergence, etc.) are swallowed so the grid
    search can continue with other orders.

    Args:
        y: 1D array of observations (e.g. the transformed series).
        order: ``(p, d, q)``.
        trend: Optional statsmodels trend code; if ``None``, inferred via ``_trend_for_order``.

    Returns:
        ``ArimaGridResult`` on success, or ``None`` if fit is invalid or raises.
    """
    p, d, q = order
    if trend is None:
        trend = _trend_for_order(d)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = ARIMA(y, order=order, trend=trend)
            res = mod.fit(method_kwargs={"warn_convergence": False, "maxiter": 500})
        aic, bic, llf = res.aic, res.bic, res.llf
        if not (np.isfinite(aic) and np.isfinite(bic)):
            return None
        conv = True
        mr = getattr(res, "mle_retvals", None)
        if isinstance(mr, dict) and "converged" in mr:
            conv = bool(mr["converged"])
        return ArimaGridResult(
            order=order,
            aic=float(aic),
            bic=float(bic),
            llf=float(llf),
            nobs=float(res.nobs),
            converged=conv,
        )
    except Exception:
        return None


def grid_search_arima(
    y: np.ndarray,
    model_type: str,
    *,
    max_p: int = 4,
    max_q: int = 4,
    max_d: int = 2,
) -> list[ArimaGridResult]:
    """
    Fit all candidate orders for a model family; sort by AIC (best first).

    Very short series are skipped (returns empty list). Each candidate is
    dropped if there are not enough points relative to ``p+d+q``.

    Args:
        y: Observations (any shape accepted by ``asarray``; non-finite removed).
        model_type: ``"AR"``, ``"MA"``, ``"ARMA"``, or ``"ARIMA"``.
        max_p: Upper bound for ``p`` in the grid.
        max_q: Upper bound for ``q`` in the grid.
        max_d: Upper bound for ``d`` (ARIMA only; ignored for AR/MA/ARMA).

    Returns:
        List of ``ArimaGridResult``, sorted by increasing AIC.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 15:
        return []

    results: list[ArimaGridResult] = []
    for order in _order_grid(model_type, max_p, max_q, max_d):
        p, d, q = order
        # Need enough points so the model is not hopelessly over-parameterised.
        if len(y) <= p + q + d + 5:
            continue
        fit = try_fit_arima(y, order)
        if fit is not None:
            results.append(fit)

    results.sort(key=lambda r: r.aic)
    return results


def fit_arima_insample(
    dates: pd.Series | np.ndarray | list,
    y: np.ndarray,
    order: tuple[int, int, int],
) -> tuple[pd.Series, pd.Series, Any] | None:
    """
    Refit ARIMA with a ``DatetimeIndex`` for plotting and residual diagnostics.

    Returns the statsmodels results object so callers can read ``.fittedvalues``
    and ``.resid``. Index alignment matters for Plotly charts.

    Args:
        dates: One timestamp per observation (same length as ``y``).
        y: Observations aligned with ``dates``.
        order: ``(p, d, q)``.

    Returns:
        Tuple ``(observed_series, fitted_series, results)``, or ``None`` if
        lengths mismatch or fitting fails.

    Note:
        ``results`` is the raw statsmodels result type (version-dependent);
        we type it as ``Any`` to avoid tight coupling.
    """
    y = np.asarray(y, dtype=float)
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(idx) != len(y):
        return None
    endog = pd.Series(y, index=idx)
    _p, d, _q = order
    trend = _trend_for_order(d)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = ARIMA(endog, order=order, trend=trend)
            res = mod.fit(method_kwargs={"warn_convergence": False, "maxiter": 500})
        fv = res.fittedvalues
        if not isinstance(fv, pd.Series):
            fv = pd.Series(np.asarray(fv, dtype=float), index=endog.index)
        else:
            fv = fv.reindex(endog.index)
        return endog, fv, res
    except Exception:
        return None
