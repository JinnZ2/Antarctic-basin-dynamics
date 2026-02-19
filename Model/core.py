import numpy as np

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
