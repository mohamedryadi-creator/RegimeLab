# Data

- `raw/` — immutable downloads/snapshots, exactly as obtained (gitignored).
- `processed/` — cleaned series produced by `regimelab.data` code (gitignored).

Nothing in this directory is versioned except this file and the directory
structure. Anything in `processed/` must be regenerable from `raw/` by library
code; the download date and source of raw data are recorded in metadata
sidecars (Phase 1).
