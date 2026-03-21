import numpy as np
import json
from pathlib import Path

# ── Constants ─────────────────────────────────────────────
PHI = 1.6180339887

# ── Parameter loading ─────────────────────────────────────

def load_parameters(path=None):
    if path is None:
        path = Path(__file__).parent / 'parameters.json'
    with open(path) as f:
        return json.load(f)

# ── Core functions ────────────────────────────────────────

def metabolic_multiplier(Q10, delta_T):
    return Q10 ** (delta_T / 10)

def adjusted_lifespan(baseline, multiplier):
    return baseline / multiplier

def trophic_energy_interception(
    primary_production,
    microbial_multiplier,
    transfer_efficiency,
    trophic_levels
):
    microbial_intercept = primary_production * (1 - 1/microbial_multiplier)
    available = (primary_production - microbial_intercept)
    for level in range(trophic_levels):
        available *= transfer_efficiency
    return available

def patch_viability(
    autocorrelation,
    body_mass,
    scaling_exponent,
    mvl_multiplier=100
):
    home_range = body_mass ** scaling_exponent
    mvl = home_range * mvl_multiplier
    effective_patch = autocorrelation ** 2 * mvl
    return effective_patch > mvl * 0.5

# ── Shared dynamics functions ─────────────────────────────

def percolation_decay(t, k_perc, t_c):
    return 1 / (1 + np.exp(k_perc * (t - t_c)))

def quadratic_warming(t, warm_a, warm_b):
    return warm_a * t + warm_b * t ** 2
