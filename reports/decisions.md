# Methodological decision log

Each entry records a decision, the options considered, and the rationale.
Keeping this log prevents silent methodology drift and feeds directly into the
final report's methodology section.

| Date | Decision | Options considered | Choice & rationale |
|---|---|---|---|
| 2026-06-10 | Repository architecture | flat scripts vs. installable library + experiments | Library (`src/` layout) with config-driven experiments, for testability and reproducibility |
| | Asset universe | — | *open* |
| | Data source & frequency | — | *open* |
| | Regime model family & K | — | *open* |
| | Validation scheme | — | *open* |
| | Cost model | — | *open (start: proportional bps + sensitivity sweep)* |
