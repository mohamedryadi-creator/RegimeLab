"""Walk-forward validation.

Open decisions (Phase 5, but the split generator is needed earlier):
- expanding window (use all history; regimes estimated on more data, but mixes
  eras) vs. rolling window (adapts faster, less data per fit);
- refit frequency (e.g. yearly vs. quarterly);
- optional embargo gap between train and test to avoid leakage through
  overlapping rolling features.

The split generator is model-agnostic: it yields (train_index, test_index)
pairs of dates, and the experiment runner is responsible for fitting regime
models only on the train slice. This is the structural defense against
look-ahead bias in model fitting.

TODO (Phase 2): implement ``walk_forward_splits(index, train_size, test_size,
scheme, embargo)`` with tests asserting that train always strictly precedes test.
"""
