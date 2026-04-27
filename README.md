# rewiring

Production-network rewiring dynamics under three anticipation modes (`aa`, `full`, `limited`). Pure-numpy implementation, in-process simulation, no file IO at the model layer.

## Layout

- `rewiring/`  -- importable package: parameter generation, network generation, equilibrium computation, simulation driver.
- `scripts/`   -- runnable scripts (figures, sweeps, studies, SLURM orchestrator).
- `tests/`     -- sanity checks (currently: boundary-conditioned partial GE).
- `results/`   -- gitignored output (figures, sweep traces).

## Quickstart

```bash
# environment: needs numpy, scipy, pandas, matplotlib, networkx
conda activate rewiring

# verify the partial-equilibrium implementation
python tests/test_partial_equilibrium.py

# produce the 3-panel AA figure for two b regimes
python scripts/plot_aa_3panel.py
```

See `CLAUDE.md` for the architecture overview, conventions, and the `run_unified_simulation` API.
