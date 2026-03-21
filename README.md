# Moments — Time-Series Explorer

Interactive **Streamlit** dashboard for exploring Bundesbank (BBK) compact-format
XML time series. The app visualises levels, first differences, and log returns;
computes descriptive moments; runs a battery of classical statistical tests; and
performs **ARIMA**-family grid search with in-sample fit diagnostics — all from
the browser.

---

## Table of contents

1. [Technical overview](#technical-overview)
   - [Project structure](#project-structure)
   - [Requirements](#requirements)
   - [Installation](#installation)
   - [Running the app](#running-the-app)
   - [Adding new series](#adding-new-series)
   - [Module reference](#module-reference)
2. [Econometric and time-series reference](#econometric-and-time-series-reference)
   - [Data transforms](#1-data-transforms)
   - [Descriptive statistics (moments)](#2-descriptive-statistics-moments)
   - [Distribution visualisation](#3-distribution-visualisation)
   - [ARIMA-family models](#4-arima-family-models)
   - [Information criteria and model selection](#5-information-criteria-and-model-selection)
   - [Statistical tests on the series](#6-statistical-tests-on-the-series)
   - [Residual diagnostics](#7-residual-diagnostics-after-arima)

---

# Technical overview

## Project structure

```
Moments/
├── app.py                  # Streamlit UI (main entry point)
├── ts_loader.py            # XML parser for Bundesbank compact SDMX
├── ts_models.py            # ARIMA grid search & in-sample refit
├── ts_tests.py             # Statistical test battery (series + residuals)
├── pyproject.toml          # Package metadata & dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme overrides
└── data/
    ├── README.md
    ├── ecb_deposit_facility_rate_eom.xml
    ├── ecb_eur_usd_reference_rate_daily.xml
    ├── ecb_main_refinancing_rate_eom.xml
    ├── ecb_marginal_lending_facility_rate_eom.xml
    ├── eonia_monthly_avg.xml
    ├── euribor_3m_monthly_avg.xml
    ├── euribor_6m_monthly_avg.xml
    ├── euribor_12m_monthly_avg.xml
    ├── german_bgb_statutory_base_rate_eom.xml
    └── german_bund_10y_yield_svensson_month_end.xml
```

## Requirements

| Dependency | Min version | Purpose |
|---|---|---|
| Python | 3.11 | Language runtime |
| Streamlit | 1.31 | Web dashboard framework |
| pandas | 2.0 | DataFrames and date handling |
| NumPy | 1.26 | Array operations |
| SciPy | 1.11 | Distributions, KDE, normality tests |
| statsmodels | 0.14 | ARIMA fitting, ADF, KPSS, Ljung-Box, etc. |
| Plotly | 5.18 | Interactive charts |

Optional dev dependency: **Ruff** (linter).

## Installation

```bash
cd Moments
python -m venv .venv          # or: uv venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .              # or: uv pip install -e .
```

## Running the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Adding new series

1. Download a Bundesbank compact-format XML file (one `<bbk:Series>` per file).
2. Drop it into `data/`.
3. (Optional) Add an English description to the `SERIES_DESCRIPTIONS` dict in
   `app.py` — the sidebar will show it.

## Module reference

| Module | Responsibility |
|---|---|
| `ts_loader.py` | Parse BBK compact XML → `(DataFrame, SeriesMeta)`. |
| `ts_models.py` | Build the `(p,d,q)` grid for AR / MA / ARMA / ARIMA, fit each candidate with `statsmodels.tsa.arima.model.ARIMA`, collect AIC/BIC/LL, refit the best model with a `DatetimeIndex` for plotting. |
| `ts_tests.py` | Run ADF, KPSS, Zivot-Andrews, Shapiro-Wilk, Anderson-Darling, K-S, Ljung-Box, BDS on the series; Ljung-Box, ARCH LM, break-variance, BDS on residuals. |
| `app.py` | Streamlit page layout, sidebar, transforms, Plotly figures, moments table, test tables, model-selection panel. |

---

# Econometric and time-series reference

Everything below explains **what the app computes** and **why**, with the
underlying formulas. Notation: \(y_t\) is the value at time \(t\), \(n\) is the
sample size, and \(\hat{\cdot}\) marks a sample estimate.

---

## 1. Data transforms

The sidebar lets you choose one of three representations. All downstream
statistics, tests, and models operate on the **transformed** series.

### 1.1 Levels (raw)

\[
x_t = y_t
\]

The original values as read from the XML file.

### 1.2 First differences

\[
x_t = \Delta y_t = y_t - y_{t-1}
\]

Removes a linear trend and is the simplest way to make an integrated series
(approximately) stationary. The first observation is lost.

### 1.3 Log returns (approximate continuously compounded returns)

\[
x_t = \Delta \ln y_t = \ln y_t - \ln y_{t-1}
\]

Only defined for \(y_t > 0\). For small changes this is close to the simple
percentage return \((y_t - y_{t-1})/y_{t-1}\). Widely used in finance because
log returns are additive over time.

---

## 2. Descriptive statistics (moments)

All formulas below use the transformed series
\(x_1, x_2, \ldots, x_n\) after dropping any non-finite values.

### 2.1 Sample mean

\[
\bar{x} = \frac{1}{n}\sum_{t=1}^{n} x_t
\]

### 2.2 Sample variance and standard deviation

\[
s^2 = \frac{1}{n-1}\sum_{t=1}^{n}(x_t - \bar{x})^2
\qquad\qquad
s = \sqrt{s^2}
\]

The denominator \(n-1\) is Bessel's correction (unbiased estimator of the
population variance).

### 2.3 Skewness (adjusted Fisher-Pearson)

\[
\hat{\gamma}_1
= \frac{n}{(n-1)(n-2)} \sum_{t=1}^{n}\!\left(\frac{x_t - \bar{x}}{s}\right)^{\!3}
\]

A symmetric distribution has \(\hat{\gamma}_1 = 0\). Positive skewness means a
longer right tail; negative skewness means a longer left tail.

### 2.4 Excess kurtosis

\[
\hat{\gamma}_2
= \frac{1}{n}\sum_{t=1}^{n}\!\left(\frac{x_t - \bar{x}}{s}\right)^{\!4} - 3
\]

(The app uses SciPy's bias-corrected version.) A normal distribution has
\(\hat{\gamma}_2 = 0\). Positive excess kurtosis ("leptokurtic") indicates
heavier tails or a sharper peak than the normal; negative ("platykurtic")
indicates lighter tails.

### 2.5 Jarque-Bera test

\[
\text{JB} = \frac{n}{6}\!\left(\hat{\gamma}_1^{\,2}
              + \frac{(\hat{\gamma}_2)^2}{4}\right)
\]

Under the null hypothesis \(H_0\!: \gamma_1 = 0 \text{ and } \gamma_2 = 0\)
(i.e. the data come from a normal distribution), the JB statistic is
asymptotically \(\chi^2(2)\). A large JB (small p-value) rejects normality.

### 2.6 Quantiles

The app reports the 25th, 50th (median), and 75th percentiles, plus the sample
minimum and maximum.

---

## 3. Distribution visualisation

### 3.1 Histogram

Observations are binned and heights are normalised to **probability density**
(area of all bars sums to 1). Bin count scales as
\(\min(60,\;\max(20,\;4\lfloor\sqrt{n}\rfloor))\).

### 3.2 Kernel density estimate (KDE)

\[
\hat{f}(x) = \frac{1}{n\,h}\sum_{t=1}^{n} K\!\left(\frac{x - x_t}{h}\right)
\]

where \(K\) is the Gaussian kernel
\(K(u) = \frac{1}{\sqrt{2\pi}}\,e^{-u^2/2}\)
and the bandwidth \(h\) is chosen by Scott's rule (SciPy default).

### 3.3 Normal reference curve

A \(\mathcal{N}(\hat{\mu},\,\hat{\sigma}^2)\) PDF is overlaid for visual
comparison, where \(\hat{\mu} = \bar{x}\) and \(\hat{\sigma} = s\).

### 3.4 Normal Q-Q plot

Theoretical quantiles \(q_i\) of \(\mathcal{N}(0,1)\) are plotted against
the order statistics \(x_{(i)}\). If the data are normally distributed, points
fall on a straight line. Heavy tails appear as an S-shaped departure. A least-
squares reference line \(x_{(i)} = a + b\,q_i\) is added.

---

## 4. ARIMA-family models

### 4.1 AR(p) — Autoregressive model of order p

\[
x_t = c + \phi_1\, x_{t-1} + \phi_2\, x_{t-2} + \cdots + \phi_p\, x_{t-p} + \varepsilon_t
\]

where \(c\) is a constant and \(\varepsilon_t \sim \text{WN}(0,\sigma^2)\) is
white noise. The current value is a linear combination of its own past values.

Using the **lag operator** \(L\) (where \(L\,x_t = x_{t-1}\)):

\[
\Phi(L)\,x_t = c + \varepsilon_t
\qquad\text{with}\quad
\Phi(L) = 1 - \phi_1 L - \phi_2 L^2 - \cdots - \phi_p L^p
\]

In the app this corresponds to **ARIMA(p, 0, 0)**.

### 4.2 MA(q) — Moving-average model of order q

\[
x_t = \mu + \varepsilon_t + \theta_1\,\varepsilon_{t-1} + \cdots + \theta_q\,\varepsilon_{t-q}
\]

The current value depends on **past forecast errors** (shocks). In lag-operator
form:

\[
x_t = \mu + \Theta(L)\,\varepsilon_t
\qquad\text{with}\quad
\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q
\]

In the app: **ARIMA(0, 0, q)**.

### 4.3 ARMA(p, q) — Combined autoregressive-moving-average

\[
\Phi(L)\,x_t = c + \Theta(L)\,\varepsilon_t
\]

An ARMA process combines the AR and MA components. It is stationary when all
roots of \(\Phi(L) = 0\) lie outside the unit circle and invertible when all
roots of \(\Theta(L) = 0\) lie outside the unit circle.

In the app: **ARIMA(p, 0, q)** with \(p \geq 1\) and \(q \geq 1\).

### 4.4 ARIMA(p, d, q) — Autoregressive integrated moving-average

If the original series is non-stationary but its \(d\)-th difference is
stationary, we write:

\[
\Phi(L)\,(1-L)^d\, y_t = c + \Theta(L)\,\varepsilon_t
\]

Here \((1-L)^d\) applies the difference operator \(d\) times:

- \(d=1\): \((1-L)\,y_t = y_t - y_{t-1}\)
- \(d=2\): \((1-L)^2 y_t = y_t - 2y_{t-1} + y_{t-2}\)

The app's grid search tries all combinations of \(p \in \{0,\ldots,p_{\max}\}\),
\(d \in \{0,\ldots,d_{\max}\}\), \(q \in \{0,\ldots,q_{\max}\}\) (excluding
the trivial (0,0,0)).

### 4.5 Trend / constant convention

| Differencing order \(d\) | `trend` parameter | Meaning |
|---|---|---|
| \(d = 0\) | `"c"` (constant) | Constant term \(c\) is estimated |
| \(d \geq 1\) | `"n"` (none) | No separate constant (differencing absorbs it) |

### 4.6 Estimation

All models are estimated by **conditional maximum likelihood** (MLE) via
`statsmodels.tsa.arima.model.ARIMA`, with up to 500 optimiser iterations.
The log-likelihood at the optimum is:

\[
\ell(\boldsymbol{\theta}) = -\frac{n}{2}\ln(2\pi\sigma^2)
    - \frac{1}{2\sigma^2}\sum_{t=1}^{n}\varepsilon_t^2
\]

where \(\boldsymbol{\theta} = (\phi_1,\ldots,\phi_p,\,\theta_1,\ldots,\theta_q,\,\sigma^2)\)
and \(\varepsilon_t\) are the one-step prediction errors conditional on past
observations.

---

## 5. Information criteria and model selection

The grid search ranks all successfully fitted models by **AIC** (lowest =
best). BIC and the log-likelihood are shown for comparison.

### 5.1 Akaike Information Criterion (AIC)

\[
\text{AIC} = -2\,\ell(\hat{\boldsymbol{\theta}}) + 2\,k
\]

where \(k\) is the number of estimated parameters (AR coefficients + MA
coefficients + variance + constant if included). AIC balances goodness of
fit against model complexity and is optimal for **prediction**.

### 5.2 Bayesian Information Criterion (BIC)

\[
\text{BIC} = -2\,\ell(\hat{\boldsymbol{\theta}}) + k\,\ln n
\]

BIC penalises additional parameters more heavily than AIC (for \(n \geq 8\))
and is consistent — i.e. it selects the true order as \(n \to \infty\) (if the
true model is in the candidate set).

### 5.3 In-sample fitted values

After selecting the best order, the model is refit with a `DatetimeIndex`.
The in-sample fitted values are:

\[
\hat{x}_t = \hat{c} + \sum_{i=1}^{p}\hat{\phi}_i\, x_{t-i}
             + \sum_{j=1}^{q}\hat{\theta}_j\,\hat{\varepsilon}_{t-j}
\]

These are plotted as "Fitted (in-sample)" against the observed series.

---

## 6. Statistical tests on the series

The **Statistical tests** tab runs the following battery on the (transformed)
series. Each test has a null hypothesis \(H_0\); rejection at level \(\alpha\)
occurs when \(p\text{-value} < \alpha\).

### 6.1 Augmented Dickey-Fuller (ADF) test

| | |
|---|---|
| **Null** | \(H_0\): The series has a **unit root** (non-stationary). |
| **Alternative** | \(H_1\): The series is stationary. |

Fits the regression:

\[
\Delta y_t = \alpha + \gamma\, y_{t-1} + \sum_{j=1}^{k}\delta_j\,\Delta y_{t-j} + u_t
\]

The test statistic is \(\hat{\gamma}\,/\,\text{se}(\hat{\gamma})\). Under
\(H_0\!: \gamma = 0\), this does **not** follow a standard \(t\)-distribution;
critical values are from Dickey-Fuller tables. The number of augmenting lags
\(k\) is chosen automatically by **AIC** (`autolag="AIC"`).

**Reject** \(H_0\) (conclude stationarity) when the test statistic is more
negative than the critical value (or the p-value is small).

### 6.2 KPSS test

| | |
|---|---|
| **Null** | \(H_0\): The series is **stationary** around a constant. |
| **Alternative** | \(H_1\): The series has a unit root. |

KPSS decomposes the series as \(y_t = \xi t + r_t + \varepsilon_t\) where
\(r_t\) is a random walk. The test statistic is:

\[
\text{KPSS} = \frac{1}{n^2\,\hat{\sigma}_\ell^{\,2}} \sum_{t=1}^{n} S_t^2
\qquad\text{with } S_t = \sum_{s=1}^{t}\hat{u}_s
\]

where \(\hat{u}_s\) are residuals from the level regression and
\(\hat{\sigma}_\ell^{2}\) is the Newey-West long-run variance estimate. The
bandwidth (number of lags) is selected automatically.

**Note:** ADF and KPSS have **opposite** nulls. Comparing them resolves
ambiguity: if ADF rejects and KPSS does not reject, stationarity is supported.

### 6.3 Zivot-Andrews test

| | |
|---|---|
| **Null** | \(H_0\): Unit root **with one structural break** in the constant. |
| **Alternative** | \(H_1\): Stationary with a break. |

The Zivot-Andrews procedure searches over all possible break dates
\(T_B \in \{2, \ldots, n-1\}\) and picks the date that gives the most
negative ADF-type \(t\)-statistic. The regression at candidate break \(T_B\) is:

\[
\Delta y_t = \alpha + \gamma\,y_{t-1} + \beta\,t + \phi\,DU_t(T_B)
             + \sum_{j=1}^{k}\delta_j\,\Delta y_{t-j} + u_t
\]

where \(DU_t(T_B) = \mathbf{1}(t > T_B)\) is a dummy for the break.
Critical values are from Zivot-Andrews tables (more negative than standard DF).

### 6.4 Shapiro-Wilk test

| | |
|---|---|
| **Null** | \(H_0\): The sample comes from a **normal** distribution. |

The test statistic is:

\[
W = \frac{\left(\sum_{i=1}^{n} a_i\, x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}
\]

where \(x_{(i)}\) are order statistics and the weights \(a_i\) come from the
expected order statistics of a standard normal sample. \(W\) is close to 1
for normal data. The app limits this to the last 5,000 observations (SciPy
recommendation).

### 6.5 Anderson-Darling test

| | |
|---|---|
| **Null** | \(H_0\): The sample comes from a **normal** distribution. |

\[
A^2 = -n - \frac{1}{n}\sum_{i=1}^{n}(2i - 1)\bigl[\ln F(x_{(i)}) + \ln(1 - F(x_{(n+1-i)}))\bigr]
\]

where \(F\) is the hypothesised CDF (here: normal with sample mean and variance).
This test gives more weight to the tails than the Kolmogorov-Smirnov test.

### 6.6 Kolmogorov-Smirnov test (standardised)

| | |
|---|---|
| **Null** | \(H_0\): The **standardised** series \(z_t = (x_t - \bar{x})/s\) follows \(\mathcal{N}(0,1)\). |

\[
D = \sup_z \bigl| F_n(z) - \Phi(z) \bigr|
\]

where \(F_n\) is the empirical CDF and \(\Phi\) is the standard normal CDF.
Because mean and variance are estimated from the same sample, the p-value is
**conservative** (a true Lilliefors test would give a smaller critical value).

### 6.7 Ljung-Box test

| | |
|---|---|
| **Null** | \(H_0\): No serial correlation up to lag \(m\). |

\[
Q(m) = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^{\,2}}{n-k}
\]

where \(\hat{\rho}_k\) is the sample autocorrelation at lag \(k\). Under
\(H_0\), \(Q(m) \sim \chi^2(m)\) for a raw series.

When applied to **ARIMA residuals**, the degrees of freedom are reduced:
\(Q(m) \sim \chi^2(m - p - q)\) (i.e. `model_df = p + q` is subtracted).
The user can choose \(m\) in the sidebar (default 10).

### 6.8 BDS test

| | |
|---|---|
| **Null** | \(H_0\): The series is **i.i.d.** (independent and identically distributed). |

The BDS statistic compares the correlation integral at embedding dimension
\(m\) with the correlation integral at dimension 1:

\[
V_m = \sqrt{n}\;\frac{C_m(\epsilon) - C_1(\epsilon)^m}{\hat{\sigma}_m}
\]

where \(C_m(\epsilon)\) is the fraction of pairs
\(\|(x_t,\ldots,x_{t+m-1}) - (x_s,\ldots,x_{s+m-1})\| < \epsilon\) in the
embedded series. Under \(H_0\), \(V_m \to \mathcal{N}(0,1)\). Rejection
signals **any** form of dependence (linear **or** non-linear), making it a
useful complement to the Ljung-Box (which only targets linear autocorrelation).
The app uses the last 500 observations and embedding dimension 2 to keep
computation manageable.

---

## 7. Residual diagnostics (after ARIMA)

After selecting the best ARIMA order, the residuals
\(\hat{\varepsilon}_t = x_t - \hat{x}_t\) are subjected to:

### 7.1 Ljung-Box on residuals

Same formula as [6.7](#67-ljung-box-test), but with
`model_df = p + q`. The adjusted degrees of freedom account for the
parameters estimated in the ARIMA model.

### 7.2 ARCH LM test (Engle, 1982)

| | |
|---|---|
| **Null** | \(H_0\): No **ARCH effects** (homoskedastic residuals). |

Regress the squared residuals on their own lags:

\[
\hat{\varepsilon}_t^2 = \alpha_0 + \alpha_1\,\hat{\varepsilon}_{t-1}^2
    + \cdots + \alpha_q\,\hat{\varepsilon}_{t-q}^2 + u_t
\]

The LM statistic is \(n R^2\) from this regression, asymptotically
\(\chi^2(q)\). The app also reports the corresponding \(F\)-statistic.
Rejection indicates **volatility clustering**: large shocks tend to be
followed by large shocks.

### 7.3 Break-variance test (Goldfeld-Quandt type)

| | |
|---|---|
| **Null** | \(H_0\): **Equal variance** in the first and second halves of the residual sample. |

The sample of residuals is split into two sub-periods and the ratio of their
variances is tested. Under \(H_0\) the ratio is approximately \(F\)-distributed.
Rejection suggests a structural break in volatility.

### 7.4 BDS on residuals

Same as [6.8](#68-bds-test), applied to \(\hat{\varepsilon}_t\). If the ARIMA
model captures all linear structure, BDS on residuals tests for remaining
**non-linear** dependence (e.g. GARCH effects, regime switching).

---

## License

This project is provided for educational and research purposes.

## Acknowledgements

Data sourced from the **Deutsche Bundesbank** time-series database
(BBK compact SDMX format).
