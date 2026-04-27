"""Rewiring: production-network rewiring dynamics under three anticipation modes.

Public submodules:
    parameters  -- random parameter generation (a, b, z) and helpers
    networks    -- SF-FA topology, AiSi productivity multipliers, helpers
    equilibrium -- general-equilibrium computation (CRS and full versions)
    partial_eq  -- tier-limited partial equilibrium (boundary-conditioned + naive)
    simulation  -- unified rewiring simulation driver (run_unified_simulation)
"""

from .parameters import EPSILON

__all__ = ["EPSILON"]
