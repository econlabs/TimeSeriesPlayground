"""
Streamlit web app: explore Bundesbank time series, moments, tests, and ARIMA models.

**What this file does (big picture)**
    1. Load one XML series from ``data/`` and optionally transform it (levels, diffs, log returns).
    2. Show charts (time series, histogram/KDE/Q-Q) and descriptive statistics.
    3. Run a grid of ARIMA-type models and display the best fit + in-sample plot.
    4. Offer a tab of formal statistical tests (stationarity, normality, etc.).

**How to run**
    From the project folder::

        streamlit run app.py

The UI state (e.g. last ARIMA search) lives in ``st.session_state``.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from ts_loader import data_dir, list_series_xml, load_bbk_series
from ts_models import ArimaGridResult, fit_arima_insample, grid_search_arima
from ts_tests import residual_test_table, series_test_table

# ---------------------------------------------------------------------------
# Theme colours (hex strings for Plotly + injected CSS)
# ---------------------------------------------------------------------------
ACCENT = "#7c3aed"  # purple — fitted line, accents
ACCENT2 = "#22d3ee"  # cyan — primary series line
BG = "#0b0f14"  # near-black main background
CARD = "#121826"  # metric card fill


def inject_css() -> None:
    """
    Push custom CSS into the page (font, gradients, sidebar contrast).

    Streamlit re-runs the script often; repeating ``st.markdown`` is the usual pattern.
    Uses ``unsafe_allow_html=True`` only for a ``<style>`` block.

    Side effects:
        Renders a markdown cell containing global CSS.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Outfit', system-ui, sans-serif;
        }}
        .stApp {{
            background: linear-gradient(165deg, {BG} 0%, #0e1520 45%, #0a1628 100%);
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #323e56 0%, #2a3448 55%, #252f42 100%);
            border-right: 1px solid rgba(124, 58, 237, 0.35);
        }}
        div[data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }}
        .block-container {{
            padding-top: 2rem;
        }}
        h1 {{
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #f8fafc 0%, {ACCENT2} 55%, {ACCENT} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def nice_label(path: Path) -> str:
    """
    Turn ``some_name.xml`` into a human title for the selectbox.

    Args:
        path: Path object pointing at a file under ``data/``.

    Returns:
        Title-cased string without the ``.xml`` suffix, underscores → spaces.
    """
    return re.sub(r"\.xml$", "", path.name).replace("_", " ").title()


# Inline CSS for the sidebar description box (Streamlit often strips ``class=`` on custom HTML).
_BLURB_PANEL_STYLE = (
    "display:block;font-size:0.82rem;font-style:italic;line-height:1.5;color:#eef2ff;"
    "padding:0.7rem 1rem;margin:0.35rem 0 1rem 0;border-radius:10px;"
    "border:1px solid rgba(124,58,237,0.45);"
    "background:linear-gradient(180deg,rgba(255,255,255,0.12) 0%,rgba(255,255,255,0.05) 100%);"
    "box-shadow:0 1px 3px rgba(0,0,0,0.35),inset 0 1px 0 rgba(255,255,255,0.07);"
)
_STRONG_STYLE = "font-style:normal;font-weight:600;color:#f8fafc;"


def blurb_to_html(text: str) -> str:
    """
    Convert markdown-style ``**bold**`` fragments to safe HTML ``<strong>``.

    Everything outside ``**...**`` is HTML-escaped to avoid injection when we
    embed the string in ``st.markdown(..., unsafe_allow_html=True)``.

    Args:
        text: Plain text with optional ``**segments**``.

    Returns:
        A single HTML fragment (no outer wrapper).
    """
    parts = re.split(r"(\*\*.+?\*\*)", text)
    out: list[str] = []
    for part in parts:
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            inner = html.escape(part[2:-2])
            out.append(f'<strong style="{_STRONG_STYLE}">{inner}</strong>')
        else:
            out.append(html.escape(part))
    return "".join(out)


# ---------------------------------------------------------------------------
# Series descriptions — loaded once from ``time_series_descriptions.txt``
# ---------------------------------------------------------------------------
_DESCRIPTIONS_FILE = Path(__file__).resolve().parent / "time_series_descriptions.txt"


def _load_series_descriptions(path: Path = _DESCRIPTIONS_FILE) -> dict[str, str]:
    """
    Parse ``time_series_descriptions.txt`` into a ``{filename: blurb}`` dict.

    File format: each entry is two consecutive non-blank lines —
    the first line is the XML filename, the second is the description text.
    Entries are separated by one or more blank lines.

    Args:
        path: Location of the descriptions file.

    Returns:
        Dict keyed by XML filename (e.g. ``"eonia_monthly_avg.xml"``).
        Empty dict if the file is missing.
    """
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    descriptions: dict[str, str] = {}
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if len(lines) >= 2:
            filename = lines[0].strip()
            blurb = " ".join(line.strip() for line in lines[1:])
            descriptions[filename] = blurb
    return descriptions


SERIES_DESCRIPTIONS: dict[str, str] = _load_series_descriptions()


def plotly_layout_kwargs() -> dict:
    """
    Shared Plotly layout options (dark template, transparent paper, font).

    Returns:
        Dict suitable for ``fig.update_layout(**plotly_layout_kwargs())``.
    """
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(18,24,38,0.6)",
        font=dict(family="Outfit, sans-serif", color="#e2e8f0"),
        margin=dict(l=48, r=24, t=48, b=48),
    )


def fig_time_series(df: pd.DataFrame, title: str, y_label: str) -> go.Figure:
    """
    Line chart of ``value`` vs ``date`` with a soft area fill under the line.

    Args:
        df: Must contain columns ``date`` and ``value``.
        title: Plot title.
        y_label: Y-axis label (often includes units).

    Returns:
        A ``plotly.graph_objects.Figure`` ready for ``st.plotly_chart``.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["value"],
            mode="lines",
            line=dict(color=ACCENT2, width=2.2),
            fill="tozeroy",
            fillcolor="rgba(34, 211, 238, 0.08)",
            name="Level",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        height=420,
        showlegend=False,
        **plotly_layout_kwargs(),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def fig_observed_vs_fitted(
    observed: pd.Series,
    fitted: pd.Series,
    order: tuple[int, int, int],
    y_label: str,
) -> go.Figure:
    """
    Overlay observed series and in-sample ARIMA ``fittedvalues``.

    Args:
        observed: Endogenous series (pandas Series, index = dates).
        fitted: Model fitted values aligned to the same index (may contain NaNs).
        order: ``(p,d,q)`` for the title string.
        y_label: Y-axis label.

    Returns:
        Plotly figure with two traces and a legend.
    """
    p, d, q = order
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed.index,
            y=observed.values,
            mode="lines",
            name="Observed",
            line=dict(color=ACCENT2, width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra>Observed</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fitted.index,
            y=fitted.values,
            mode="lines",
            name="Fitted (in-sample)",
            line=dict(color=ACCENT, width=2.2, dash="dot"),
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra>Fitted</extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"In-sample fit · ARIMA({p},{d},{q})", font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        height=440,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **plotly_layout_kwargs(),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    return fig


def fig_density(values: np.ndarray, title: str, x_label: str) -> go.Figure:
    """
    Histogram (density-normalised) plus KDE and a normal PDF with sample μ, σ̂.

    Args:
        values: 1D array of observations.
        title: Plot title.
        x_label: X-axis label.

    Returns:
        Plotly figure.
    """
    fig = make_subplots(rows=1, cols=1)
    # Bin count: scale gently with sample size.
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=min(60, max(20, int(np.sqrt(len(values))) * 4)),
            histnorm="probability density",
            name="Empirical",
            marker=dict(color=ACCENT, line=dict(width=0)),
            opacity=0.55,
        )
    )
    if len(values) >= 5:
        kde = stats.gaussian_kde(values)
        xs = np.linspace(values.min(), values.max(), 256)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kde(xs),
                mode="lines",
                name="KDE",
                line=dict(color=ACCENT2, width=3),
            )
        )
        # Compare to N(μ̂, σ̂) using sample moments (beginner-friendly reference curve).
        mu, sigma = float(np.mean(values)), float(np.std(values, ddof=1))
        if sigma > 0:
            ys = stats.norm.pdf(xs, mu, sigma)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="Normal (fitted)",
                    line=dict(color="#f472b6", width=2, dash="dash"),
                )
            )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=x_label,
        yaxis_title="Density",
        barmode="overlay",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **plotly_layout_kwargs(),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def fig_qq(values: np.ndarray) -> go.Figure:
    """
    Normal Q–Q plot: ordered sample vs normal theoretical quantiles.

    Uses ``scipy.stats.probplot`` with ``fit=False`` so we get raw quantile pairs,
    then plots them as **arrays** (not ``osm[0]`` scalars—common gotcha).

    Args:
        values: 1D sample.

    Returns:
        Plotly figure with points and a reference line from linear regression on quantiles.
    """
    osm, osr = stats.probplot(values, dist="norm", fit=False)
    osm = np.asarray(osm, dtype=float).ravel()
    osr = np.asarray(osr, dtype=float).ravel()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(size=6, color=ACCENT2, opacity=0.75),
            name="Sample",
        )
    )
    slope, intercept, _, _, _ = stats.linregress(osm, osr)
    line_x = np.array([float(osm.min()), float(osm.max())])
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=slope * line_x + intercept,
            mode="lines",
            line=dict(color="#f472b6", width=2),
            name="Reference",
        )
    )
    fig.update_layout(
        title=dict(text="Normal Q-Q plot", font=dict(size=16)),
        xaxis_title="Theoretical quantiles",
        yaxis_title="Ordered values",
        height=360,
        showlegend=False,
        **plotly_layout_kwargs(),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def compute_moments(x: np.ndarray) -> dict[str, float]:
    """
    Basic descriptive moments + Jarque–Bera normality test.

    Args:
        x: Numeric array (non-finite entries dropped).

    Returns:
        Dict of scalar metrics, or empty dict if fewer than 2 finite points.
    """
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return {}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, bias=False))  # excess kurtosis (normal → 0)
    jb_stat, jb_p = stats.jarque_bera(x)
    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "variance": std**2,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "jb_statistic": float(jb_stat),
        "jb_pvalue": float(jb_p),
    }


def main() -> None:
    """
    Build the full Streamlit UI: sidebar, tabs, model section, session state.

    This function is long on purpose—everything visible runs top-to-bottom each
    Streamlit rerun. Keep widget keys stable when you add new inputs.
    """
    # --- Page setup ---
    st.set_page_config(
        page_title="Moments · Time series",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # Discover XML files; bail out early if the data folder is empty.
    xml_files = list_series_xml()
    if not xml_files:
        st.error(f"No `*.xml` files found in `{data_dir()}`. Add series under `data/`.")
        st.stop()

    st.title("Time series & moments")
    st.caption("Bundesbank compact data · distribution, KDE, and descriptive statistics")

    # ------------------------------------------------------------------
    # Sidebar: pick series, show blurb, pick transform
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Series")
        labels = [nice_label(p) for p in xml_files]
        choice = st.selectbox(
            "Choose a series",
            options=range(len(xml_files)),
            format_func=lambda i: labels[i],
        )
        path = xml_files[choice]

        blurb = SERIES_DESCRIPTIONS.get(path.name)
        if blurb:
            st.markdown(
                f'<div style="{_BLURB_PANEL_STYLE}" role="note">{blurb_to_html(blurb)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "_No extended description for this file._ "
                "Add an entry in `time_series_descriptions.txt`."
            )

        st.markdown("### Transform")
        transform = st.radio(
            "Values to analyze",
            options=["Levels", "First differences", "Log returns (approx.)"],
            index=0,
            help="Log returns use diff(log(level)), dropping non-positive levels.",
        )

        st.markdown("---")
        st.caption("Data folder: `data/`")

    # Load raw series from disk (always the un-transformed XML content first).
    df_raw, meta = load_bbk_series(path)

    # Apply the user's transform to build the working DataFrame ``df`` and numpy ``values``.
    if transform == "Levels":
        df = df_raw.copy()
        y_label = meta.unit or "Value"
        series_name = "Level"
    elif transform == "First differences":
        df = df_raw.copy()
        df["value"] = df["value"].diff()
        df = df.dropna(subset=["value"])
        y_label = f"Δ ({meta.unit or 'value'})"
        series_name = "First difference"
    else:
        # Log-return needs strictly positive levels before taking log.
        df = df_raw[df_raw["value"] > 0].copy()
        df["value"] = np.log(df["value"]).diff()
        df = df.dropna(subset=["value"])
        y_label = "Δ log(level)"
        series_name = "Log return"

    values = df["value"].to_numpy(dtype=float)

    # If series or transform changed, forget the last ARIMA grid (avoid stale fits).
    data_sig = (str(path), transform, int(len(values)))
    if st.session_state.get("_data_sig") != data_sig:
        st.session_state["_data_sig"] = data_sig
        for k in ("arima_grid", "arima_family"):
            st.session_state.pop(k, None)

    # Header under the title: official English title + metadata line.
    st.markdown(f"#### {meta.title_en}")
    sub = f"`{meta.bbk_id}` · {meta.freq or '?'} · {meta.unit or ''}"
    if meta.unit_mult and meta.unit_mult != "0":
        sub += f" (×10^{meta.unit_mult})"
    st.caption(sub)

    m = compute_moments(values)
    if not m:
        st.warning("Not enough observations after transform.")
        st.stop()

    # Quick numeric summary row (Streamlit ``metric`` widgets).
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observations", f"{int(m['n']):,}")
    c2.metric("Mean", f"{m['mean']:.4f}")
    c3.metric("Std. dev.", f"{m['std']:.4f}")
    c4.metric("Skewness", f"{m['skewness']:.3f}")
    c5.metric("Excess kurtosis", f"{m['excess_kurtosis']:.3f}")

    # ------------------------------------------------------------------
    # Tabs: visuals, moments table, formal tests
    # ------------------------------------------------------------------
    tab_ts, tab_dist, tab_stats, tab_tests = st.tabs(
        ["Time series", "Distribution", "Moments & tests", "Statistical tests"]
    )

    with tab_ts:
        st.plotly_chart(
            fig_time_series(
                df,
                title=f"{series_name} over time",
                y_label=y_label,
            ),
            use_container_width=True,
        )

    with tab_dist:
        c_left, c_right = st.columns((1.1, 0.9))
        with c_left:
            st.plotly_chart(
                fig_density(
                    values,
                    title="Histogram + KDE (+ normal fit)",
                    x_label=y_label,
                ),
                use_container_width=True,
            )
        with c_right:
            st.plotly_chart(fig_qq(values), use_container_width=True)
            st.caption(
                "Q-Q plot: heavy tails vs normal show as S-curve; compare with Jarque-Bera below."
            )

    with tab_stats:
        st.subheader("Descriptive statistics")
        stats_df = pd.DataFrame(
            [
                {"Statistic": "Minimum", "Value": m["min"]},
                {"Statistic": "25% quantile", "Value": float(np.percentile(values, 25))},
                {"Statistic": "Median", "Value": m["median"]},
                {"Statistic": "75% quantile", "Value": float(np.percentile(values, 75))},
                {"Statistic": "Maximum", "Value": m["max"]},
                {"Statistic": "Variance", "Value": m["variance"]},
                {"Statistic": "Jarque–Bera statistic", "Value": m["jb_statistic"]},
                {"Statistic": "Jarque–Bera p-value", "Value": m["jb_pvalue"]},
            ]
        )
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn("Value", format="%.6g"),
            },
        )
        st.info(
            "**Jarque-Bera** tests joint null of skewness = 0 and excess kurtosis = 0 (normality). "
            "Low p-value suggests non-normality (common for financial series)."
        )

        with st.expander("Raw preview (last rows)"):
            st.dataframe(df.tail(12), use_container_width=True)

    with tab_tests:
        st.subheader("Tests on the current transform")
        st.caption(
            "All tests use the **same series** as the charts (after sidebar transform). "
            "**ADF** and **KPSS** have opposite nulls—compare them carefully. "
            "**Phillips–Perron** is not exposed in this statsmodels build; **Zivot–Andrews** adds a one-break unit-root view."
        )
        lj_lag = st.number_input(
            "Ljung–Box lag",
            min_value=5,
            max_value=30,
            value=10,
            key="lb_lag_main",
            help="Used for Ljung–Box on the series and (when available) on ARIMA residuals.",
        )
        ser_df = series_test_table(values, ljung_lag=int(lj_lag))
        if ser_df.empty:
            st.warning("Not enough data to run tests.")
        else:
            st.dataframe(ser_df, use_container_width=True, hide_index=True)

        st.markdown("#### Residual diagnostics (after ARIMA)")
        st.caption(
            "Fits the **best ARIMA** from your last **Find best orders** run (section below). "
            "If you changed series or transform, run the search again."
        )
        arima_stored = st.session_state.get("arima_grid")
        fam_stored = st.session_state.get("arima_family")
        if (
            isinstance(arima_stored, list)
            and len(arima_stored) > 0
            and fam_stored
        ):
            best_o = arima_stored[0]
            p0, d0, q0 = best_o.order
            st.markdown(f"Using **{fam_stored}** best order **ARIMA({p0},{d0},{q0})**.")
            fit_r = fit_arima_insample(df["date"], values, best_o.order)
            if fit_r is None:
                st.warning("Could not refit ARIMA for residual tests.")
            else:
                _obs, _fv, res_fit = fit_r
                resids = np.asarray(res_fit.resid, dtype=float)
                res_df = residual_test_table(resids, best_o.order, ljung_lag=int(lj_lag))
                if res_df.empty:
                    st.warning("Not enough residual observations for diagnostics.")
                else:
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.info(
                    "**Ljung–Box** on residuals uses `model_df = p + q`. **ARCH LM** targets volatility clustering. "
                    "**Break-variance** compares variance in early vs late subsamples of residuals."
                )
        else:
            st.info("Run **Find best orders** below to populate residual-based tests.")

    # ------------------------------------------------------------------
    # Model selection (full width below tabs)
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("## Model selection")
    st.caption(
        "Full-width panel below the charts · statsmodels ARIMA grid, best specification by **lowest AIC**."
    )

    m_left, m_right = st.columns([2.2, 1], gap="large")
    with m_left:
        model_family = st.radio(
            "Process",
            options=["AR", "MA", "ARMA", "ARIMA"],
            index=2,
            horizontal=True,
            help="AR: ARIMA(p,0,0). MA: ARIMA(0,0,q). ARMA: ARIMA(p,0,q), p,q≥1. ARIMA: full (p,d,q) with d≥0.",
        )
        p1, p2, p3 = st.columns(3)
        with p1:
            max_p = st.number_input("Max p", min_value=1, max_value=8, value=4, help="Max AR order")
        with p2:
            max_q = st.number_input("Max q", min_value=1, max_value=8, value=4, help="Max MA order")
        with p3:
            max_d = 0
            if model_family == "ARIMA":
                max_d = st.number_input(
                    "Max d",
                    min_value=0,
                    max_value=3,
                    value=2,
                    help="Max differencing order (ARIMA only).",
                )
            else:
                st.caption("Max **d** applies when **ARIMA** is selected.")

        run_search = st.button("Find best orders", type="primary", use_container_width=False)

    with m_right:
        st.markdown("**Series (fit target)**")
        st.caption(meta.title_en[:220] + ("…" if len(meta.title_en) > 220 else ""))
        st.markdown(
            f"`{meta.bbk_id}` · **{meta.freq or '—'}** · {meta.unit or '—'}  \n"
            "Sidebar **transform** applies to plots **and** this search."
        )

    # Button click triggers a fresh grid search and stores results in session_state.
    if run_search:
        with st.spinner("Fitting candidate models…"):
            arima_results: list[ArimaGridResult] = grid_search_arima(
                values,
                model_family,
                max_p=max_p,
                max_q=max_q,
                max_d=max_d,
            )
        st.session_state["arima_grid"] = arima_results
        st.session_state["arima_family"] = model_family

    # Read back stored results (may be from a previous rerun).
    results_raw = st.session_state.get("arima_grid")
    fam_done = st.session_state.get("arima_family")
    results: list[ArimaGridResult] = results_raw if isinstance(results_raw, list) else []

    if results and fam_done == model_family:
        best = results[0]
        p, d, q = best.order
        st.success(f"**Best:** ARIMA({p},{d},{q})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AIC", f"{best.aic:.2f}")
        m2.metric("BIC", f"{best.bic:.2f}")
        m3.metric("Log-likelihood", f"{best.llf:.2f}")
        m4.metric("n (effective)", f"{int(best.nobs):,}")
        st.caption(f"Optimiser converged: **{best.converged}**")
        top = results[: min(12, len(results))]
        rank_df = pd.DataFrame(
            [
                {
                    "Rank": i + 1,
                    "Order (p,d,q)": f"({r.order[0]},{r.order[1]},{r.order[2]})",
                    "AIC": r.aic,
                    "BIC": r.bic,
                    "LL": r.llf,
                    "OK": r.converged,
                }
                for i, r in enumerate(top)
            ]
        )
        st.markdown("**Top candidates**")
        st.dataframe(
            rank_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "AIC": st.column_config.NumberColumn(format="%.2f"),
                "BIC": st.column_config.NumberColumn(format="%.2f"),
                "LL": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.markdown("### Fitted vs observed")
        st.caption(
            "One-step / in-sample **fittedvalues** from the best model (same transform as above)."
        )
        fit_plot = fit_arima_insample(df["date"], values, best.order)
        if fit_plot is None:
            st.warning("Could not refit the best model for plotting — try other orders or a transform.")
        else:
            obs_s, fitted_s, _res = fit_plot
            st.plotly_chart(
                fig_observed_vs_fitted(obs_s, fitted_s, best.order, y_label),
                use_container_width=True,
            )
    elif fam_done is not None and fam_done != model_family:
        st.warning("Model class changed — click **Find best orders** again to refresh.")
    elif results_raw is None:
        st.info("Set **Max p**, **Max q** (and **Max d** for ARIMA), then click **Find best orders**.")
    elif len(results) == 0:
        st.warning(
            "No successful fits — try smaller **Max p / q / d** or a different **transform**."
        )


if __name__ == "__main__":
    main()
