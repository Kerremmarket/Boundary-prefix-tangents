# Obtaining and preparing the data

## Access

Obtain an account with access to **OptionMetrics IvyDB US**, for example through your institution's WRDS subscription. A WRDS account by itself does not establish access to every vendor product. Ask your institution's WRDS administrator if IvyDB US is unavailable. Official sources: [OptionMetrics on WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/), [WRDS access methods](https://wrds-www.wharton.upenn.edu/pages/about/3-ways-use-wrds/).

## Sample and fields

The extraction script targets SPX, OptionMetrics **secid 108105**, from **January 2005 through August 2025**. For each calendar month, it selects the last observed SPX option-quote date, giving 248 monthly dates in the original extraction. This is not a selection of calendar month-end dates regardless of trading activity. Option expiries are restricted to **30–730 calendar days** after the quote date.

The script discovers the live `optionm_all` schema and records the table/field mapping; it does not assume that every subscriber has identical physical table names. It extracts:

| Dataset | Required content |
|---|---|
| Option prices | Date, expiry, option identifier, symbol, put/call flag, strike, bid/offer, volume, open interest, implied volatility, Greeks; available settlement, root, contract-size and related metadata |
| Standardized volatility surface | Date, maturity in days, put/call flag, delta, implied volatility |
| Forward prices | Date, expiry or maturity, settlement indicator, forward price |
| Zero curve | Date, maturity in days, continuously compounded rate |
| Underlying prices | Date and SPX closing level |

The canonical column mappings and optional fields are defined in `ROLE_SPECS` in `research/financial_extension/src/wrds_optionmetrics_extract.py`. The script preserves raw strike and adds normalized strike, with a guard for the vendor's strike scaling. Rate conversion from percentage to decimal occurs in the preparation routines. Do not pre-convert rates or normalize strikes a second time.

## Extraction

From the repository root, after the main dependency installation:

```bash
python -m pip install -r requirements-wrds.txt
export PAPER1_OPTIONMETRICS_OUTPUT_DIR="$PWD/data/raw"
python research/financial_extension/src/wrds_optionmetrics_extract.py --schema-only
python research/financial_extension/src/wrds_optionmetrics_extract.py
```

Authenticate interactively with your own WRDS account. The script declines the optional prompt to create a `.pgpass` file. It writes resumable yearly checkpoints, a schema inventory, monthly-date list and extraction report locally. Re-running resumes completed checkpoints; `--force` explicitly re-queries them. A changed vendor schema may require inspection of the recorded mapping.

The following files must exist in `data/raw/`:

```text
spx_option_prices_monthly_2005_2025.parquet
spx_vol_surface_monthly_2005_2025.parquet
spx_forward_prices_monthly_2005_2025.parquet
spx_zero_curve_monthly_2005_2025.parquet
spx_spot_monthly_2005_2025.parquet
```

The filenames retain `2005_2025`; the extractor's `END_DATE` is **2025-08-31**. It does not include September–December 2025.

## Local preparation

```bash
python reproduce.py prepare --data-dir data/raw
```

This command:

1. Computes one-year ATM IV, the one-year 25-delta skew, the 182-to-730-day ATM IV difference, and the interpolated one-year rate. It selects 12 dates by deterministic farthest-point sampling after robust standardization, without using regret outcomes.
2. Cleans the quote intervals, reconciles settlement/forward conventions, selects eligible expiry slices, repairs call-price shape constraints and calibrates the annual lognormal mixture using the original routines.
3. Builds the annual panel and exports the first five selected dates for the cliquet/Bermudan experiments: **2016-10-31, 2008-11-28, 2020-02-28, 2009-04-30, 2006-03-31**, in that selection order. A changed first-five selection stops preparation with an explicit error.
4. Records local hashes and compares the cliquet input hashes with the historical ones. It does not rewrite the original experiment configuration to claim a fresh download was the original frozen input.

The annuity diagnostics additionally use **2022-10-31, 2023-05-31 and 2020-03-31**. The union of the dates used by these financial experiments is eight dates. However, reproducing the market-state selection itself requires the full 248-date universe, not only those eight dates.

Calibration uses the nearest eligible one-year expiry, at least 12 strikes, and a maximum expiry distance of 120 days. The annual-panel runner records failed calibrations explicitly and exits unsuccessfully if any calibration fails. The cleaning and calibration source and synthetic tests specify the remaining rules exactly.

## Fresh downloads versus the original snapshot

Vendor corrections and differences in numerical libraries can change results. Preserve your extraction report and local run records. Identical file hashes provide a stronger check than matching rounded table values; unequal hashes require examination and do not by themselves identify the cause.

The public repository contains no original observations, frozen market-derived inputs, or market-output archives. Results are generated locally. Redistribution of those local files is governed by the applicable data licence.
