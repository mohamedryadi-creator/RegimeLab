# Experiments

Runnable experiment scripts. Each script:

1. takes a config file from `configs/` (and nothing else — no hardcoded parameters);
2. calls only `regimelab` library code;
3. writes results (tables, figures, fitted models) to `experiments/outputs/<name>/`.

Outputs are gitignored; an experiment is reproducible from its config, the code
version, and the seed.
