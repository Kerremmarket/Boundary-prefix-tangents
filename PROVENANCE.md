# Public source export

The underlying research code is drawn from scientific revision `f460994b7fe15495374c6bec7e3fa8aacb0eb69c` and its companion reproducibility archive. This identifier belongs to the original research repository, not the history of this public repository. The accompanying manuscript has subsequently undergone editorial and mathematical exposition revisions. No manuscript source or outdated paper PDF is included here.

## Preserved and adapted components

The theoretical examples, numerical kernels, calibrators, data extraction/cleaning/selection routines, experiment configurations, and numerical assertions are retained from the scientific baseline. `SOURCE_ORIGIN.json` records the original export hashes and identifies the files adapted for this release.

Three runner files adapt checks that depended on private Git history or on redistributing frozen market inputs:

- `research/cliquet_feasibility/src/run_pilot.py`: replaces historical Git ancestry/tree checks with the public source manifest; accepts inputs verified against the local preparation record while retaining the original input hashes in the experiment configuration.
- `research/cliquet_integration/src/run_uncapped_audit.py`: checks the public source manifest and registered locally regenerated predecessor outputs in place of the old private tree/hash requirement.
- `research/prefix_comparator_revision/src/run_experiments.py`: checks public source integrity and locally registered inputs/predecessor outputs instead of requiring descent from the original private Git commit.

These adaptations do not change pricing, optimization, tangent coefficients, experiment designs, numerical tolerances, or comparison rules. The actual public Git revision is recorded by the original runners. Original protocol flags and commit identifiers describe historical design timing; they do not assert that a new reader's run preceded the original outcomes.

`reproduce.py` provides the execution order, converts the regenerated panel to the existing solver input schemas, and copies generated numerical figure PDFs to the location expected by the original output validator. `reproduction_support.py` provides public-source and local-artifact integrity checks. Locally registered files must remain unchanged between preparation and use. Registration records the origin of a local calculation; it is not an independent correctness certificate or a claim of equality with the historical archive.

## Export exclusions

No raw observations, frozen market-derived inputs, financial result archives, account credentials, manuscript drafts, internal reviews, or old package-level manifests are distributed. The historical cliquet package-manifest test is replaced by public-source integrity checks because its manifest covered the original private data/result bundle. Synthetic numerical tests are retained, and the original financial output assertions remain available after regeneration.

Historical methodological protocols are included for transparency. One author-machine absolute path in the historical provenance document is replaced with a descriptive relative path; that editorial substitution does not alter the recorded dates or protocol history.

## Validation scope

See `VALIDATION.md` for the checks performed on this public export. A successful synthetic test run is not a full reproduction of the financial experiments, and a new WRDS download is not guaranteed to be byte-identical to the original vendor snapshot.
