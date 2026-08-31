# WTI Crude Oil Spot and Futures — Data for Cointegration Application

**Retrieved:** 2026-07-08 (via WebFetch only; no API key used)
**Built by:** `build_and_diagnose.py` (numpy/pandas; scipy/statsmodels unavailable in sandbox, so ADF / Johansen trace / ARCH-LM implemented directly)

## Files
- `wti_spot_futures.csv` — aligned monthly panel, columns: `date, spot, F1, F2, F3, F4, l_spot, l_F1..l_F4` (`l_` = natural log).
- `raw/spot_MCOILWTICO.txt` — raw FRED spot series (date value).
- `raw/RCLC{1,2,3,4}m.txt` — raw EIA futures tables (year + 12 monthly values, `NA` = no data).
- `build_and_diagnose.py` — reproducible build + diagnostics script.

## Sources
| Series | Description | Source | Endpoint |
|---|---|---|---|
| spot | WTI (Cushing, OK) spot, $/bbl, monthly (MCOILWTICO) | EIA via FRED | `https://fred.stlouisfed.org/data/MCOILWTICO.txt` |
| F1–F4 | Cushing WTI NYMEX futures contracts 1–4, $/bbl, monthly (RCLC1–RCLC4) | EIA | `https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RCLC{n}&f=M` |

Note: the `fredgraph.csv?id=MCOILWTICO` CSV endpoint returned as undecodable binary through WebFetch; the plain-text table endpoint `.../data/MCOILWTICO.txt` was used instead (identical data).

## Coverage
- FRED spot: 1986-01 to 2026-06 (486 obs).
- EIA futures F1–F4: end **2024-04**. EIA discontinued the NYMEX futures price series; the source page states "Futures prices after April 5, 2024, are not available." F1/F3 begin 1983, F2/F4 begin 1985.
- **Aligned sample (all five series present): 1986-01 to 2024-04, 460 monthly observations.** This is the sample in the CSV.

## Transformations
- Frequency: monthly throughout (native EIA/FRED monthly averages of daily prices; no aggregation performed by us).
- Log prices `l_*` = natural log of the level. All aligned-sample prices are strictly positive (min spot = 11.35 in 1998-12), so logs are well defined.

## Negative-price issue (Apr 2020)
The −$37.63 WTI print on 2020-04-20 was a single daily front-month settlement. At **monthly** frequency the averages stay positive (2020-04: spot 16.55, F1 16.70), so the log transform is unaffected. Had daily data been used, April 2020 would require handling (drop the day, use a monthly average, or an inverse-hyperbolic-sine transform). Monthly averaging is the chosen and sufficient handling here.

## Volatility regimes (spot monthly log-returns)
Full-sample sd = 0.098. Sub-period sd: 2007–09 = 0.125; 2014–16 = 0.106; 2020 = 0.309. Largest monthly moves: 2020-03 (−0.55), 2020-04 (−0.57), 2020-05 (+0.55), 1990-08 Gulf War (+0.39), 2008-Q4 GFC crash (−0.31, −0.29, −0.33), 2014-12 (−0.25). Strong evidence for conditional heteroskedasticity / regime shifts — relevant for the GRRR (generalized reduced-rank) application.

## Diagnostics summary (see script output)
- **ADF:** log levels non-stationary (fail to reject unit root, trend model); first differences stationary (reject at 1%). All five series I(1).
- **Johansen trace (VAR lag k=2, restricted constant):** reject r≤0,1,2,3; fail to reject r≤4. **Cointegration rank = 4 = p−1**, i.e. one common stochastic trend (the price level) and 4 stationary basis relations — as expected.
- **Basis relations** (l_Fi − l_spot): ADF rejects unit root for all i (stationary), consistent with rank p−1.
- **ARCH-LM (q=4) on VAR(2) residuals:** strong ARCH in l_spot (p≈2e-5) and l_F1 (p≈5e-6); weaker/insignificant for l_F2–F4. Confirms conditional heteroskedasticity, motivating volatility-robust cointegration inference.
