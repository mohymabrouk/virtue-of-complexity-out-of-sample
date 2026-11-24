# Project 2: does the virtue of complexity survive outside its sample?

Written 2026-11-21, before any model code was written and before any result was looked at.
Everything in sections 1-9 is the pre-registered design. Any later change goes into the
"Changes during the build" log at the bottom, with the reason.

## 1. Question

Kelly, Malamud & Zhou (KMZ), "The Virtue of Complexity in Return Prediction", *Journal of Finance* 79(1), 2024,
459-503 (NBER w30217). Claim: a market-timing model built on thousands of random Fourier features (RFF) plus
ridge regression, trained on only 12 months, has a higher out-of-sample Sharpe ratio than simple models, even
though its out-of-sample R² is often negative.

We test it where the authors could not:

- **Part A, after the paper's sample.** Replicate the US result on the Welch-Goyal predictors over the paper's
  own sample, then score the exact same specification once on the months after the paper's sample ends.
- **Part B, across countries.** Run the same specification on non-US equity markets from the
  Jensen-Kelly-Pedersen (JKP) global factor data.

## 2. The paper's design (read from the NBER w30217 PDF, section 6 and appendix G)

| Item | KMZ (as written in the paper) |
|---|---|
| Target | Monthly return of the CRSP value-weighted index (Nagel 2025 confirms it is the *excess* return) |
| Return scaling | Returns divided by their trailing 12-month standard deviation ("volatility standardised"). Target, trained-on returns and the traded asset are all this standardised return; "the market" means the standardised market everywhere |
| Predictors | 15: dfy, infl, svar, de, lty, tms, tbl, dfr, dp, dy, ltr, ep, b/m, ntis (Welch-Goyal) and one lag of the market return (footnote 32) |
| Predictor scaling | Divided by an expanding-window historical standard deviation; 36 months are needed first, so the analysis sample starts in 1930 |
| RFF map | S_i,t = P^-1/2 [sin(γ ω_i'G_t), cos(γ ω_i'G_t)], ω_i ~ iid N(0, I_15), eq. (21) |
| Bandwidth | γ = 2 (footnote 35) |
| Feature scaling | Training RFFs and the test RFF are divided by their standard deviation in the training sample (footnote 36) |
| Estimator | Ridge without intercept: β̂(z) = (zI + T⁻¹ Σ S_t S_t')⁻¹ T⁻¹ Σ S_t R_t+1 (section 3.1) |
| Training window | Rolling T = 12 months (main); 60 and 120 shown too |
| Training pairs at t | (R_t, S_t-1), ..., (R_t-T+1, S_t-T); forecast β̂'S_t for R_t+1 |
| P grid | "P ∈ {2, ..., 12,000}", the first P of 12,000 RFFs; c = P/T up to 1000 |
| z grid | log10 z ∈ {-3, -2, -1, 0, 1, 2, 3} |
| Seeds | 1,000 independent RFF draws; performance statistics are averaged across draws |
| Strategy | Position π_t = forecast; strategy return π_t R_t+1 |
| Metrics | OOS R² (1 - MSE / mean(R²), zero benchmark, section 2.3), Sharpe, alpha, alpha t-stat and information ratio from a regression on the (standardised) market |
| Sample | CRSP 1926-2020; t ∈ {T, ..., 1091}, i.e. OOS months 1931-01 to 2020-12 (1,080 months; Nagel's appendix also says "1931-2020") |

**Replication target (pre-registered).** Table 1, Panel A, "Nonlinear", T = 12, c = 1000 (P = 12,000),
z = 10³: Sharpe **0.47** (t = 4.5), IR vs market 0.31 (t = 2.9), R² 0.6%. Primary number: the Sharpe 0.47,
computed the paper's way (per-seed Sharpe, averaged over seeds). Tolerance: **±0.10** absolute, about one
standard error of an annual Sharpe over 90 years. Secondary (reported, not tested): alpha t-stat 2.9, R² 0.6%,
the linear z = 10³ Sharpe 0.46, the linear ridgeless Sharpe -0.11.

## 3. Critiques (verified from the sources themselves)

- **Nagel, S. (2025), "Seemingly Virtuous Complexity in Return Prediction", NBER Working Paper 34104.**
  PDF read. When P ≫ T the RFF forecast is a kernel-weighted average of the 12 training returns. In short
  windows the weights mostly reflect recency, and they shrink when predictor volatility is high, so the strategy
  is a *volatility-timed momentum* rule. His rule (eq. 14): position = 0.05 · (1/σ̄²_x) · Σ_{k=0}^{11} (12-k)/78 · r_t-k,
  where σ̄²_x is the average within-window variance of the 15 predictors. As a second factor it removes about
  2/3 of KMZ's alpha (Table I). He also reproduces KMZ's Figure 8 t-stat (2.81) with their code.
- **KMZ themselves (appendix G)** compare VoC with a 12-month momentum position mean₁₂/var₁₂ and find VoC
  alpha of 5.8% a year (t = 3.0) over it.
- **Buncic, D. (2025), "Simplified: A Closer Look at the Virtue of Complexity in Return Prediction",
  Stockholm University working paper.** Not read directly (no copy found); cited only through Nagel (2025),
  who reports that Buncic shows the rising performance-complexity curve disappears when forecasts are averaged
  across RFF draws *before* performance is computed. We test this ourselves by reporting both conventions.

Consequences for this project: add Nagel's volatility-timed momentum and KMZ's momentum as baselines, and report
both the paper's averaging convention ("mean over seeds") and the ensemble convention ("average the forecast
across seeds, then evaluate").

## 4. Data (free, no keys; raw files cached in `~/.cache/quant-research/p2/raw/`, never re-downloaded)

| Source | URL | File | Content |
|---|---|---|---|
| Welch-Goyal predictors, 2025 vintage | Amit Goyal's site https://sites.google.com/view/agoyal145, link "All data up to 2025 (includes data from GW (2008) and GWZ (2024) in one file)" → `https://drive.google.com/uc?export=download&id=17mw_IpaiLFDrGnrPRQ2o1ugV5nJsZuD1` | `goyal/PredictorData2025_gdrive.xlsx`, sheet `Monthly` | 1871-01 to **2025-12** |
| US market excess return | `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip` | `french/F-F_Research_Data_Factors_CSV.zip` | Mkt-RF, RF, 1926-07 to 2026-09 (built from the 202609 CRSP file) |
| JKP market returns, all countries | `https://data.jkpfactors.com/public/%5Ball_countries%5D_%5Bmkt%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip` | `jkp/all_countries_mkt_monthly_vw_cap.zip` | capped value-weighted market excess return in USD, 93 countries, to 2025-12 |
| JKP 13 theme returns, all countries | `https://data.jkpfactors.com/public/%5Ball_countries%5D_%5Ball_themes%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip` | `jkp/all_countries_all_themes_monthly_vw_cap.zip` | 13 theme factor returns per country, to 2025-12 |

All downloaded on 2026-11-21 with a generic User-Agent; no login needed (the URL scheme comes from the
download form on jkpfactors.com/data and its `availability.json` manifest). The `download` stage writes a
manifest with URL, size, SHA-256 and date.

**US predictor construction** (Welch & Goyal 2008 definitions, from the raw columns of the 2025 vintage):
dp = log d12 - log price; dy = log d12 - log price₋₁; ep = log e12 - log price; de = log d12 - log e12;
svar; b/m; ntis; tbl; lty; ltr; tms = lty - tbl; dfy = BAA - AAA; dfr = corpr - ltr;
infl lagged one extra month (CPI release delay, as in Welch-Goyal); lag mkt = Mkt-RF of month t.
The forecast target is French Mkt-RF (CRSP value-weighted excess return), the free stand-in for KMZ's CRSP index.

**Post-sample test period.** The last predictor month is 2025-12, so the last forecast is for 2026-01.
Post-sample OOS months: **2021-01 to 2026-01 = 61 months** (about 5 years). With 61 months the standard error
of an annual Sharpe is about 0.44, so a true IR of 0.31 has only ~17% power to show up as significant at 5%
(one-sided). We report this plainly. The sub-window after the JF publication (2024-01 to 2026-01, 25 months)
is shown only for information.

## 5. The pre-registered specification

- **T = 12**, **γ = 2**, RFF pairs [sin, cos] of γ ω'x with ω ~ N(0, I_K).
- **P grid (20 values):** 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 36, 48, 60, 120, 240, 600, 1200, 3000, 6000,
  12000 (nested: the first P features of one 12,000-feature draw). c = P/12 from 0.17 to 1000.
- **z grid (7 values):** 10⁻³ ... 10³. VoC grid = 140 cells.
- **Seeds:** 1,000 for Part A if a run fits in about 20 minutes on 2 cores, otherwise fewer (recorded).
  Part B: as many as fit in about 20 minutes for all countries (expected 20-50; recorded).
- **Predictor scaling:** x_s / sd(x_first..x_s), expanding, including month s, at least 36 observations; no demeaning.
- **Return scaling:** R_s = r_s / sd(r_s-12 .. r_s-1) (12 months, ddof = 1). Every forecast at month t uses
  only data up to t.
- **Feature scaling:** in each window, divide the 12 training feature rows and the test row by the training
  std of each feature (ddof = 0, numpy default). The P^-1/2 factor of eq. (21) cancels, so it is dropped.
- **Estimator:** ridge without intercept, solved in dual form: forecast = s_t' S' (S S' + zT I)⁻¹ R, a 12×12
  system per month (equal to the primal β̂(z) above; tested).
- **Two evaluation conventions:** (a) *paper* ("mean over seeds"): a statistic per seed, averaged over seeds;
  (b) *ensemble*: forecasts averaged over seeds, then one strategy is evaluated. The replication target uses (a),
  as the paper does. Inference (CIs, bootstrap, deflated Sharpe) uses (b), because only (b) is a single strategy
  an investor could run. Both are reported.

## 6. Baselines (all trade the same standardised market return R_t+1)

1. **Buy-and-hold**: π = 1 (this is already a volatility-managed market: raw position 1/σ_t).
2. **Historical mean**: π = expanding mean of R up to t.
3. **Linear ridge (kitchen sink)**: the 15 standardised predictors, same window, same within-window scaling,
   no intercept; z = 0+ (minimum-norm least squares, KMZ's "ridgeless") and z = 10⁻³ ... 10³. With T = 12 < 15
   predictors plain OLS is not identified, so z = 0+ is the kitchen sink.
4. **TSMOM sign**: π = sign(Σ_{k=0}^{11} R_t-k) (Moskowitz, Ooi & Pedersen 2012).
5. **Mom12**: π = mean(R_t-11..t) (equal weights; the limit of VoC when predictors are static, KMZ appendix G).
6. **KMZ momentum**: π = mean₁₂(r)/var₁₂(r) on raw excess returns (KMZ appendix G).
7. **Nagel vol-timed momentum**: π = (1/σ̄²_x,t) Σ_{k=0}^{11} (12-k)/78 · R_t-k, with σ̄²_x,t the mean
   variance of the scaled predictors over the 12 training months (eq. 14; the 0.05 scale is irrelevant for
   Sharpe/t-stats and is dropped). We use the standardised returns R (the VoC training targets) so that the rule
   is the same object the kernel weights act on.

A separate "1/trailing variance" rule is not added: buy-and-hold of R already holds 1/σ_t of the raw market.

## 7. Statistics

`sharpe` and `deflated_sharpe` are *copied* (not imported) from Alpha Audit
(`C:/dev/quant-agent-lab/backend/src/qal/lab/metrics.py`), with an attribution comment.

- Annualised Sharpe = mean/sd·√12, with a standard error from Lo (2002) with the Mertens (2002) non-normal
  correction: Var(SR) = (1 + SR²/2 - γ₃SR + (γ₄-3)/4·SR²)/n (monthly SR), annualised; 95% CI = ±1.96 SE.
- Deflated Sharpe (Bailey & López de Prado 2014): trials = all strategy variants reported for that period
  (140 VoC cells + 8 linear-ridge cells + 5 simple rules = 153); trial Sharpe variance = variance of the
  per-period Sharpe across those 153 series.
- Alpha and information ratio from an OLS regression on the standardised market: t-stat from iid OLS (KMZ's
  convention; t ≈ IR·√years) and Newey-West (12 lags) in the CSVs.
- OOS R² = 1 - Σ(R - f)² / Σ R².
- ΔSharpe vs buy-and-hold with a 95% circular block-bootstrap CI over months (2,000 resamples, block length
  round(n^(1/3)), same resampled months for both series).

## 8. Protocol

- Stages: `download`, `build`, `replicate` (paper sample only; the data are cut at 2020-12 before the model
  sees them), `test` (post-sample, run once), `global` (Part B), `report`.
- `test` scores 2021-01..2026-01 once and appends to `results/test_log.jsonl` the UTC timestamp, the config
  hash, the code hash and summary numbers. A second run needs `--force` and is logged as a forced rerun.
- No parameter is chosen on any evaluation data: T, γ, grids and seeds come from the paper; the baselines are
  fixed above. The full complexity curve (every P × z cell) is reported; no "best cell" is reported alone.
- Main-table VoC rows: z = 10³ (the paper's main z) at c = 1, 10, 100, 1000.

## 9. Part B design (fixed before looking at any result)

- **Asset:** JKP `mkt`, capped value-weighted, monthly excess return in USD, per country.
- **Predictors (15):** the 13 JKP theme returns of month t, the market excess return of month t, and the
  market's trailing 12-month return standard deviation at t. Same scaling as the US case.
- **Sample per country:** the longest run of consecutive months with the market return, all 13 themes and at
  least 10 stocks in the market portfolio; 36 months of burn-in for predictor scaling, 12 for training; the
  country is kept if at least 120 OOS months remain. A data check (no model output) gives 42 locations: the US
  and 41 other countries. The US with JKP predictors is reported separately as a bridge, not in the non-US
  distribution.
- **Same specification** as Part A for every country: T, γ, P grid, z grid; no per-country tuning.
- **Reported:** cross-country distribution (median, quartiles) of Sharpe, ΔSharpe vs buy-and-hold, alpha t,
  R²; counts of countries where VoC (P = 12,000, z = 10³) beats buy-and-hold, linear ridge z = 10³, Nagel's
  vol-timed momentum and TSMOM, with a binomial sign-test p-value; a pooled result: an equal-weighted portfolio
  across available countries each month of each strategy, evaluated like a single strategy (Sharpe, CI,
  alpha vs the equal-weighted buy-and-hold).

## 10. Compute plan

Per seed, build the RFF matrix (N × 12,000) once; for every month the 13 window rows are divided by the training
std and turned into a 13×13 Gram matrix per P block (nested P is a cumulative sum over feature blocks);
then a batched eigendecomposition of the 12×12 training Gram gives all 7 z at once. Work per seed is
about N·13²·12,000 multiply-adds (≈ 2·10⁹ for the US), so 1,000 seeds is plausible on 2 cores. Per-seed
statistics are accumulated on the fly; only seed-averaged forecasts are stored. Runs longer than ~2 minutes go to
the background with a log file.

## 11. Tests (pytest)

1. Dual ridge = primal ridge on small random problems (several z, P < T and P > T).
2. RFF map: shape, sin/cos pairing, nested-P property.
3. Leak test: replacing all data after month t leaves every forecast up to t unchanged (VoC, linear, simple
   rules), including the predictor and return scaling.
4. Timing-strategy return on a toy series.
5. Copied `sharpe` and `deflated_sharpe` equal the Alpha Audit functions on fixed inputs (the original
   functions are loaded from their source text without importing the package).
6. Replication target within ±0.10 of 0.47, or `xfail` with the gap explained in the README.

## 12. Risks and fallbacks

- **Data vintage / index differences** (French Mkt-RF vs CRSP VW, Goyal 2025 vintage vs 2021 vintage, the
  inflation lag, log vs level ratios): may move the replication by a few hundredths. Recorded as deviations.
- **Compute:** if 1,000 seeds do not fit, cut seeds for the full grid (record), keep P = 12,000 and z = 10³.
- **JKP blocked:** would fall back to Ken French's international country returns; not needed (download works).
- **Short post-sample window:** 61 months gives low power; reported with the power calculation, no extension.

## 13. Deviations from the paper (known now)

1. Market return: Ken French Mkt-RF instead of CRSP VW pulled from WRDS (same index family, free).
2. Predictors from the 2025 vintage of Goyal's file (data revised since KMZ's vintage).
3. Inflation lagged one month (Welch-Goyal convention; KMZ do not say).
4. P grid has 20 points; the paper's exact grid between 2 and 12,000 is not listed.
5. Feature std uses ddof = 0 (equivalent to multiplying z by 12/11 vs ddof = 1).

## Changes during the build

(appended as they happen)

### 2026-11-21, during the thin slice (paper sample only; no post-sample or Part B model output seen)

1. **Seeds cut.** One seed of the full 20 × 7 grid over 1931-2020 takes 1.6 s alone and 2.2-2.8 s while three
   other agents share the laptop (2 cores for this project). 1,000 seeds would take 30-45 minutes, above the
   20-minute budget. **Part A uses 300 seeds, Part B 30 seeds per country.** The same seeds are used for the
   paper sample and the post-sample test. At P ≥ 1,200 the seed-to-seed spread of the Sharpe is small, so
   this mainly adds noise to the low-P "mean over seeds" curve.
2. **Panel start.** The US panel starts in 1926-07 (first CRSP month); expanding predictor std are computed
   from there (ntis starts in 1926-12). Earlier Goyal data are not used.
3. **Replication gap found, diagnosed, spec kept.** The pre-registered spec gives a per-seed Sharpe of about
   0.30 at P = 12,000, z = 10³ (3 seeds) against KMZ's 0.47, with alpha t ≈ 1 against 2.9. Diagnostics, all on
   the paper sample (scripts in `scripts/diag_*.py`):
   - the dual engine equals a direct primal ridge on the same windows (`diag_primal.py`);
   - our data reproduce Nagel's (2025, Table I) volatility-timed momentum: alpha t 3.60 and IR 0.383 against
     his 3.684 and 0.388 (`diag_nagel.py`), so the predictors and returns match theirs closely;
   - on raw (unstandardised) excess returns, our RFF ridgeless P = 12,000 strategy has alpha t 2.70 and IR 0.287
     against Nagel's 2.417 and 0.255 obtained with KMZ's own replication code (`diag_raw.py`);
   - the gap therefore comes from the return standardisation step. Trailing std over t-12..t-1 or a
     root-mean-square instead of a std does not close it; a std window that includes the forecast month
     (look-ahead, not causal) moves the linear numbers onto KMZ's (ridgeless Sharpe -0.111 vs -0.11; z = 10³
     alpha t 3.12 vs 3.1) but not the VoC alpha t (1.9 vs 2.9), so it is not adopted
     (`diag_retscale.py`). Uncentred RFF scaling (Nagel's formula (11)) and an unlagged inflation series do not
     close the gap either (`diag_variants.py`).
   The primary spec is unchanged (it follows the paper's text and is causal). Test 6 is therefore expected to
   be `xfail`, with this explanation in the README.
4. **Secondary specification added: raw returns (Nagel's convention).** Because the raw-return version
   reproduces a published number from KMZ's own code, every Part A and Part B result is also reported with
   the training target, the traded asset and the benchmark all equal to the raw excess return (no volatility
   standardisation of returns; predictors and RFF unchanged). The simple rules use raw returns in this
   variant (so `nagel_volmom` is exactly Nagel's eq. 14 up to scale). This is fixed now, before the post-sample
   test and before any Part B run. The standardised spec stays the primary one.
5. **Deflated Sharpe trials** stay at 153 per period and per specification.

### 2026-11-21, runs

6. `replicate` (300 seeds, 892 s): Sharpe 0.310 at the target cell (paper 0.47) → outside tolerance; test 6
   marked `xfail(strict=True)`; a second, separately stated check (raw convention vs Nagel's KMZ-code numbers,
   ±0.5 on t, ±0.06 on IR) added and passing.
7. `test` ran once at 2026-11-21T22:44:57Z (config hash `c737047adb5c40f4`, code hash `eb3e44ded8f11321`),
   61 months 2021-01..2026-01, 58 s. No rerun.
8. `global` started right after, 30 seeds per country, 42 locations.
9. Code changed after the test only in reporting files (`report.py`, `readme.py`, `readme_text.py`) and a
   `scripts/replication_diagnostics.py` that re-runs the paper-sample diagnostics into
   `results/replication_diagnostics.json` so the README can quote them from a results file.
10. `global` finished in 1,113 s: 42 locations (41 non-US + the US bridge), 30 seeds each.
11. `scripts/replication_diagnostics.py` re-ran the paper-sample diagnostics (3 draws per variant) into
    `results/replication_diagnostics.json`; numbers agree with the exploratory runs above.
12. Reporting additions decided after seeing the results (descriptive only, no new model runs): a post-sample
    year-by-year table of the target cell, and a US bridge table that puts the Welch-Goyal model (Part A positions)
    next to the JKP-predictor US model on the same 1956-10..2025-12 months.
13. 2026-11-22, after publication on GitHub: `scripts/external_replication_check.py` reads the only public KMZ
    replication we found (nkonts, commit 40ab1c8; data only, none of its code executed) and writes
    `results/external_replication_check.json`. It standardises returns with `returns.rolling(12).std()` and targets
    `returns.shift(-1)`, i.e. the look-ahead window of item 4 in the README's Replication section, and reports a
    Sharpe of 0.42 at P = 12,000, z = 10³ (500 draws). README only: a headline paragraph and Replication item 5.
    No model, test or result of this project changed. The intro line no longer refers to a series, and the
    post-sample cumulative figure moved under the headline.

### 2026-11-24, research note (paper-sample diagnostics and one post-sample score of construction B; no Part B run)

14. **Pre-declared before running: a second external replication check.** A literature and code search for the note
    found a third public KMZ replication, rrikrv/voc-replication (commit 9c77321, 2026-09-28). Its README reports
    KMZ's Table 1 with a *causal* return scaling (r_t divided by the uncentred RMS of r_{t-12..t-1}): linear
    z = 10³ Sharpe 0.46 / IR 0.33 (paper 0.46 / 0.33) and RFF P = 12,000, z = 10³ Sharpe 0.495 / IR 0.33
    (paper 0.47 / 0.31). Its RFF, within-window feature scaling, ridge (z·T in dual form) and window alignment are
    the same as ours (read, not executed). Our RMS variant gives 0.35, so the remaining difference must be in the
    data. Diagnostic, data only (none of its code is executed, its CSV is not committed; SHA-256 recorded):
    `scripts/diag_external_data.py` compares its committed `data/voc_data.csv` with our panel column by column,
    then runs our unchanged engine on (a) its data, (b) our data, and (c) hybrids that swap one block at a time
    (returns and return scaling; predictors), P = 12,000, z ∈ {10⁻³, 10³}, 1931-01..2020-12 targets, 10 draws
    each (seeds 0-9). Output: `results/external_data_diagnostic.json`. Whatever it shows is reported in the note
    and README as is; the primary specification and every frozen result stay unchanged.
15. **Result of 14 (10 draws each, `results/external_data_diagnostic.json`):** our unchanged engine on rrikrv's
    data gives VoC (P = 12,000, z = 10³) Sharpe 0.49, alpha t 3.06, IR 0.33 and linear z = 10³ Sharpe 0.46,
    alpha t 3.06, IR 0.33, ridgeless -0.11: KMZ's Table 1 without a look-ahead volatility window (one of its
    data choices is not real-time, see 17). On our data the same engine gives 0.31. Swapping only its returns in gives 0.36, only its predictors 0.40. **So the replication gap comes from our
    data construction, not from the timing of the volatility window; the look-ahead explanation in the README is
    withdrawn.** The engine, the frozen primary specification and every frozen result are unchanged.
16. **Pre-declared before running: construction B and its post-sample test.** Construction B rebuilds rrikrv's data
    choices from our own raw files (Goyal 2025 file, to 2025-12):
    (i) each predictor divided by its expanding std from its first month in the Goyal file (not from 1926-07);
    (ii) inflation not lagged an extra month; (iii) the market return and the lagged-market predictor are Goyal's
    `ret - Rfree` (S&P 500 with dividends) instead of Ken French's Mkt-RF; (iv) returns divided by the uncentred
    RMS of the previous 12 months instead of their std. Everything else (RFF, γ = 2, T = 12, within-window
    scaling, ridge, seeds) is the frozen specification.
    - `scripts/construction_b.py` first checks that its builder with all four switches off reproduces our dataset
      exactly, then adds the switches one at a time, cumulatively in the order (i)-(iv), on 1931-01..2020-12,
      10 draws each, P = 12,000, z ∈ {10⁻³, 10³}.
    - Then construction B with draws 0-299: the paper-sample replication (per-draw statistics averaged, as KMZ)
      and **one post-sample score on 2021-01..2025-12 (60 months; the Goyal file ends in 2025-12)**, for the
      seed-averaged forecast (as in the README's main table) and per draw. On the same 60 months, the frozen
      construction A is recomputed with the same draws for a like-for-like comparison (its official 61-month
      result stays the one in `results/test_log.jsonl`). Buy-and-hold and the linear ridge z = 10³ are reported
      for both constructions. Scored once; output `results/construction_b.json`.
    - Primary/secondary labels do not change: construction A stays the pre-registered primary; B is reported as
      "the construction that reproduces the paper".
17. **Result of 16 (`results/construction_b.json`, scored once, finished 2026-11-24T10:35:28+00:00).** The builder with all four
    switches off reproduces our dataset exactly. Cumulative on 1931-2020 (10 draws), VoC P = 12,000, z = 10³:
    A 0.31 → (i) 0.34 → (ii) 0.38 → (iii) 0.42 → (iv) = B 0.49; alpha t 1.0 → 3.1. B equals the result on the
    external file, so the four switches account for the whole difference. With 300 draws B gives 0.49, alpha t 3.1,
    IR 0.33 (A: 0.31, the same as the official replication). Steps (i), (iii) and (iv) use only past data; step (ii)
    does not, because CPI for month t is released during month t+1 (about two weeks after the forecast is made).
    **Post-sample, 2021-01..2025-12 (60 months), seed-averaged forecast: B -0.06 (SE 0.45, alpha t -0.4) against
    0.71 for its buy-and-hold; A -0.01 against 0.57 on the same months.** Linear ridge z = 10³: B 0.45, A 0.30.
    The construction that reproduces the paper also loses to buy-and-hold after its sample. Reported in the README
    (headline, Replication item 6, a dated correction of the look-ahead paragraph, a Related-work section) and in
    the research note `paper/note.pdf`, whose numbers are written from results/ by `python -m voc.note`.
