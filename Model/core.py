import numpy as np

# Functions below the divider were added in the 2026-08
# literature review. Each has a neutral parameter value at
# which it reduces to the model's prior behaviour, so
# existing simulations are unaffected until a mechanism is
# deliberately engaged. See Docs/literature.md.


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


# ---------------------------------------------------------
# 2026-08 literature review additions
# ---------------------------------------------------------


def accelerating_temperature(baseline, t, linear_rate, acceleration=0.0):
    """Warming trajectory with a quadratic term.

    T = baseline + linear_rate*t + acceleration*t^2

    Observed abyssal warming in the Antarctic sector is not
    linear: the eastern Bellingshausen Basin trend for
    2017/18-2023/24 is roughly triple the trend measured
    since 1992/95 (Johnson et al., GRL 2024). Setting
    acceleration=0 recovers the linear ramp used previously.

    Literature: Docs/literature.md section 1.
    """
    t = np.asarray(t, dtype=float)
    return baseline + linear_rate * t + acceleration * t ** 2


def dynamic_transfer_efficiency(
    baseline_efficiency,
    delta_T,
    sensitivity_per_C,
    floor=0.01
):
    """Trophic transfer efficiency declining with warming.

    Seven years of experimental warming reduced trophic
    transfer efficiency by up to 56% at +4 C relative to
    ambient (Barneche et al., Nature 592:76-79, 2021).
    The default sensitivity of 0.14 per C is a first-order
    linear fit to that endpoint.

    Efficiency enters trophic_energy_interception() once per
    trophic level, so a given per-step reduction compounds
    across the chain. That compounding is the point.

    Three caveats, stated because the number looks more solid
    than it is: the experiment was freshwater mesocosms, "up
    to 56%" is an upper bound rather than a mean, and a
    linear fit to a single endpoint is a crude form. Setting
    sensitivity_per_C=0 recovers the fixed-efficiency model.

    Literature: Docs/literature.md section 3.
    """
    delta_T = np.asarray(delta_T, dtype=float)
    reduced = baseline_efficiency * (1 - sensitivity_per_C * delta_T)
    return np.clip(reduced, floor, baseline_efficiency)


def oxygen_availability(
    delta_T,
    baseline_saturation=1.0,
    committed_loss_fraction=0.10,
    reference_delta_T=4.0
):
    """Normalised dissolved oxygen under warming.

    Committed ocean oxygen loss is roughly fourfold what has
    already been realised, and about 80% of it lands below
    2000 m; the deep ocean loses more than 10% of its
    pre-industrial oxygen content even under immediate
    emissions cessation (Nature Communications, 2021).

    Loss is scaled linearly against reference_delta_T, which
    represents the warming over which the committed loss is
    taken to be expressed. This is a normalisation choice,
    not a measured relationship.

    Literature: Docs/literature.md section 4.
    """
    delta_T = np.asarray(delta_T, dtype=float)
    loss = committed_loss_fraction * (delta_T / reference_delta_T)
    return np.clip(baseline_saturation - loss, 1e-6, None)


def metabolic_index(
    oxygen,
    metabolic_multiplier_value,
    body_mass=1.0,
    mass_sensitivity=0.25
):
    """Ratio of oxygen supply to temperature-dependent demand.

    Follows the Metabolic Index framing of Deutsch et al.
    (Science, 2015).

    Output is normalised, not an absolute aerobic scope: with
    the defaults, unit body mass at zero warming returns
    exactly 1.0 by construction. Read relative decline under
    warming and ordering across body masses. Do not read the
    absolute level as a viability threshold -- that would
    require species-specific critical O2 data this model does
    not carry.

    The mass term implements the finding that oxygen
    availability and body mass jointly modulate ectotherm
    responses to warming, rather than the constraint being
    mass-neutral (Nature Communications, 2023). Larger bodies
    are penalised more sharply. Setting mass_sensitivity=0
    removes the asymmetry.

    The direction of the mass effect is empirical; the
    exponent is not. Sweep it.

    Literature: Docs/literature.md sections 4 and 5.
    """
    demand = metabolic_multiplier_value * body_mass ** mass_sensitivity
    return oxygen / demand


def percolation_connectivity(
    t,
    threshold_time,
    steepness,
    base_autocorrelation=1.0,
    floor=0.0
):
    """Sigmoidal connectivity collapse.

    Replaces exponential decay with a percolation-style
    transition: connectivity holds, then drops sharply near
    threshold_time.

    ac(t) = base / (1 + exp(steepness * (t - threshold_time)))

    Motivated by Antarctic sea ice, which held a range for
    four decades and then stepped to a new state. Statistical
    analysis over 1979-2022 identifies a regime break in
    September 2016, followed by the February 2023 record low
    of 1.77 million km2 -- 36% below the 1979-2022 mean
    minimum (Communications Earth & Environment, 2023;
    NOAA Climate.gov). An exponential decay cannot produce
    that shape at any parameter value.

    Three caveats.

    Sea ice extent is a proxy for habitat continuity, not a
    measurement of it. The mapping is assumed.

    "Percolation threshold" in the sea ice physics literature
    refers to brine flow through the ice matrix -- a different
    phenomenon at a different scale that happens to share this
    functional form. It is not evidence for this use.

    The bare sigmoid asymptotes to zero, whereas the observed
    sea ice step was to a lower state rather than to nothing.
    floor > 0 represents residual connectivity after the
    transition. It defaults to 0.0 to match the form proposed
    in additions.md, which is almost certainly too severe at
    long horizons.

    Implements additions.md item 1.
    Literature: Docs/literature.md section 6.
    """
    t = np.asarray(t, dtype=float)
    z = np.clip(steepness * (t - threshold_time), -700, 700)
    sigmoid = 1.0 / (1 + np.exp(z))
    return floor + (base_autocorrelation - floor) * sigmoid


def mass_dependent_connectivity(
    autocorrelation,
    body_mass_scaling,
    sensitivity=1.0
):
    """Connectivity penalty scaled by body mass.

    connectivity_factor = ac ** (sensitivity * body_mass_scaling)

    Large-bodied strategies are penalised under warming
    through at least three partly independent channels: home
    range demand under fragmentation, oxygen supply limits
    under deoxygenation (Nature Communications, 2023), and
    direct size-at-maturity reduction (PNAS, 2025).

    The response is a tendency with real variance, not a law
    -- in one broad analysis roughly 45% of species were
    larger, not smaller, in warmer water. Setting
    sensitivity=1.0 with body_mass_scaling=0.5 recovers the
    symmetric ac**0.5 used previously.

    Implements additions.md item 2.
    Literature: Docs/literature.md section 5.
    """
    autocorrelation = np.clip(np.asarray(autocorrelation, dtype=float), 0.0, None)
    return autocorrelation ** (sensitivity * body_mass_scaling)


def maintenance_adjusted_lifespan(baseline, multiplier, decoupling=0.0):
    """Lifespan compression damped by active maintenance.

    adjusted_lifespan() assumes rate-of-living: longevity as a
    passive consequence of slow metabolism. Greenland shark
    genomics argues against the pure form -- chromosome-level
    assembly recovers expansions in TNF, TLR and LRRFIP
    (NF-kB pathway) alongside DNA repair and cancer
    resistance signatures (PNAS, 2026), and cardiac tissue
    shows resilience rather than age-related degeneration
    (Chiavacci et al., Aging Cell, 2026).

    Encoded maintenance of that kind is not obviously a
    function of ambient temperature, so warming should damp
    lifespan rather than divide it.

    decoupling in [0, 1]:
        0.0 -> identical to adjusted_lifespan()
        1.0 -> lifespan independent of metabolic rate

    No study provides a compression coefficient for this
    lineage. The default of 0.5 in parameters.json is a
    deliberate mid-point that makes the uncertainty visible.
    It is a knob to sweep, not a result.

    Literature: Docs/literature.md section 2.
    """
    multiplier = np.asarray(multiplier, dtype=float)
    effective = multiplier ** (1.0 - decoupling)
    return baseline / effective
