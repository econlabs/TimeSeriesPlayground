# Curated time series (copied from `raw/`)

Bundesbank compact SDMX XML. Each file is one series (`bbk:Series` + `bbk:Obs` with `TIME_PERIOD`).

| File | What it is | Rough span (check last `TIME_PERIOD` in file) |
|------|------------|-----------------------------------------------|
| `euribor_3m_monthly_avg.xml` | **EURIBOR** 3-month, monthly average | From **1999-01** (euro area) |
| `euribor_6m_monthly_avg.xml` | EURIBOR 6-month, monthly average | From **1999-01** |
| `euribor_12m_monthly_avg.xml` | EURIBOR 12-month, monthly average | From **1999-01** |
| `eonia_monthly_avg.xml` | **EONIA** (overnight index), monthly average | From **1999-01** (ECB from 4 Jan 1999) |
| `ecb_main_refinancing_rate_eom.xml` | **ECB** main refinancing operations rate, month-end | From **1999-01** |
| `ecb_deposit_facility_rate_eom.xml` | ECB **deposit facility** rate, month-end | From **1999-01** |
| `ecb_marginal_lending_facility_rate_eom.xml` | ECB **marginal lending** facility rate, month-end | From **1999-01** |
| `ecb_eur_usd_reference_rate_daily.xml` | **EUR/USD** ECB reference rate, **daily** | Long history (pre-1999 Frankfurt spot; see series notes in XML) |
| `german_bund_10y_yield_svensson_month_end.xml` | German **Bund** term structure, **10y** residual maturity (Svensson), **month-end** | Long history (see `BBK_COMM_GEN` in XML) |
| `german_bgb_statutory_base_rate_eom.xml` | German **BGB §247 statutory base rate**, month-end | From **2002** (replaces DÜG base rate for many contracts) |

**Note:** The `raw` library did not surface classic **USD LIBOR** or BBA **EURIBOR** filenames as separate series; the closest standard money-market benchmarks here are **EURIBOR** (EMMI), **EONIA**, and **ECB standing facility / MRO** rates—all commonly used in finance and econometrics.

Original paths are preserved in the XML `BBK_ID` / `Header` `ID` fields.
