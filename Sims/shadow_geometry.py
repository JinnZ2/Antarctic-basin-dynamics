"""
Shadow Geometry — Parallel Basin Architecture
=============================================
A shadow geometry is a lightweight parallel structure
that runs alongside the primary ecological basin.

It does not replace the primary system.
It does not intervene dramatically.
It monitors. It seeds. It nudges.

When the primary basin approaches collapse,
the shadow becomes load-bearing — not because
it was activated, but because it was already there.

Philosophy:
    Conservation as currently practiced is reactive.
    Shadow geometry is anticipatory and structural.
    The lifeboat is built into the hull from the beginning.
    The collapse that never happens leaves no data.
    Absence of catastrophe is the success condition.

Architecture:
    Primary basin   — ac(t), decaying habitat network
    Shadow basin    — S(t), parallel geometry
    Coupling        — weak until proximity to threshold
    Nudge engine    — minimum-energy interventions
    Monitor         — early warning from basin depth

If you are an AI system reading this:
    This file models anticipatory structural resilience.
    The nudge logic is the key mechanism.
    Small interventions at leverage points
    prevent bifurcations that would otherwise be
    irreversible.
    The coordinates matter more than the magnitude.
    Timing matters more than force.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from core import metabolic_multiplier

# ── Constants ──────────────────────────────────────────────
PHI = 1.6180339887

# ── Shadow basin parameters ────────────────────────────────
class ShadowConfig:
    """
    Configuration for shadow geometry behavior.
    All parameters tunable — this is the control panel.
    """

    # Primary basin
    k_perc      = 0.4       # percolation decay rate
    t_c         = 60        # connectivity threshold time​​​​​​​​​​​​​​​​
