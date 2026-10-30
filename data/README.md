# Data

## Included

`vix_futures_cfe/` — CBOE VIX futures (VX) daily settlements, 171 contract files,
January 2013 to June 2026. Public CBOE data. Loaded by `vix_lab.load_vx_cfe`.

## Not included

`bid_ask_quotes/` — SPX and SPXW end-of-day option chains, 2010-2023, about 1.7 GB
across 1,072 files. Distributed by optionsDX (https://www.optionsdx.com) and not
redistributed here. One file per month; each row is a contract with its quote timestamp,
expiry, strike, bid, ask and the underlying index level.

The free release ends at December 2023, which is why the option sample stops there. The
cut-off is a constraint, not a choice: no year was dropped after inspecting results.

To rebuild the curve from the raw archive, place the monthly files under
`data/bid_ask_quotes/` and run the extraction in `ts_extract.py`. The result is
`ts_reinvert.parquet`, which is already present at the repository root, so this step is
only necessary if you want to verify the construction itself.
