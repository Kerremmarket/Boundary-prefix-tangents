# Boundary-Prefix Tangents

Research code for **Boundary-Prefix Tangents and Second-Order Regret in Multidate Optimal Stopping**, by **Erdinç Akyıldırım, Shaen Corbet, and Kerem Çiftçi**.

Corresponding author: Kerem Çiftçi, Department of Economics, Bogazici University, Istanbul, Turkey. Email: kerem.ciftci@std.bogazici.edu.tr. [ORCID: 0009-0007-9080-1643](https://orcid.org/0009-0007-9080-1643).

This repository provides the numerical implementation, experiment configurations, data extraction and calibration scripts, and tests. **OptionMetrics observations and frozen market-derived inputs are not distributed here.** Readers obtain OptionMetrics IvyDB US through their own licensed access and generate the inputs locally. See [DATA_ACCESS.md](DATA_ACCESS.md) for the precise sample and workflow.

## Setup and tests without market data

Use Python 3.12. The numerical dependencies match the recorded research environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py tests
python reproduce.py theory
```

The tests cover the tangent examples, option pricing, quote cleaning, calibration, date selection, boundary solvers, and interaction decomposition using synthetic inputs. They do not require a WRDS account. These tests do not constitute proofs of the manuscript's theorems.

## Reproduce the market experiments

First follow [DATA_ACCESS.md](DATA_ACCESS.md) to download the five required datasets to `data/raw/`. From the repository root:

```bash
python reproduce.py prepare --data-dir data/raw
python reproduce.py cliquet
python reproduce.py prefix
```

`prepare` reconstructs the 248-date selection/calibration panel and the five-snapshot cliquet inputs. `cliquet` regenerates the original numerical baseline and its uncapped-state/refinement audit. `prefix` regenerates the complete-one-date versus boundary-prefix comparisons, tables, and figures used in the revised paper. Run the stages in this order, in separate processes.

For the additional financial diagnostics:

```bash
python reproduce.py bermudan
python reproduce.py annuity
```

These two stages require `prepare`, but do not depend on `cliquet` or `prefix`. The original routines contain substantial numerical refinement and simulation; allow ample time and memory. The prefix protocol records a 45-minute/3-GiB resource bound, not a runtime guarantee for every machine. The annuity diagnostics include grids up to 120,000 points; the Bermudan diagnostics include multiple training and evaluation seeds.

The numerical output paths and the revised manuscript's reported comparisons are explained in [RESULTS.md](RESULTS.md). Generated inputs, figures, and results are ignored by Git.

## Provenance and interpretation

The scientific baseline is source revision `f460994b7fe15495374c6bec7e3fa8aacb0eb69c` from the original research repository. That historical revision is not a commit in this public repository. Mathematical kernels and experiment configurations are preserved. Three runner integrity checks have been adapted for a fresh public checkout; see [PROVENANCE.md](PROVENANCE.md).

`SOURCE_MANIFEST.sha256` checks the public release files. `local-reproduction.json` records locally generated inputs and intermediate results. `input-comparison.json` reports whether the newly prepared cliquet inputs match the original frozen hashes. A different hash can reflect CSV serialization, numerical libraries, or a changed vendor snapshot; it is not silently labelled an exact reproduction. Output validators check the numerical identities and the paper's expected dispositions and will fail if those checks differ.

Historical protocols in `docs/historical-protocols/` describe the sequence of the original research, including preliminary exploration. Their paths and Git identifiers refer to the original research checkout. Use the commands above for this public export.

## Data access and reuse

The market data originate from [OptionMetrics IvyDB US via WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/). Access is subject to the provider's subscription and licence conditions. This repository does not provide credentials or redistribute those datasets. Do not commit downloaded observations or generated market inputs.

No open-source licence has yet been designated for this code; publication of this repository does not itself grant an additional software licence. Contact the corresponding author for reuse permissions beyond applicable law. The OptionMetrics data remain governed separately by their provider's terms.
