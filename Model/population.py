"""Age-structured population dynamics for the slow integrator.

Implements additions.md item 4 and closes the recruitment gap
recorded in Docs/literature.md section 7.

Two reasons this had to stop being a scalar.

The observed signal is in recruitment. The clearest ecological
change reported in Antarctic mid-trophic species is weak
juvenile recruitment with a concomitant rise in mean body
length -- a structure signal that a single abundance number
cannot carry.

Ecological memory needs a better definition. geometry.md
operationalises memory as the lifespan of dominant long-lived
species. Recent work cautions explicitly against reading
sensitivity to environmental change off life-history position,
and against treating the fast-slow continuum as one axis. A
projection matrix gives defensible measures instead:
generation time (the integration window), the damping ratio
(how fast a perturbation is forgotten), and the period of the
transient oscillation. None of them is lifespan, and they do
not respond to warming the way lifespan does.

Demography is anchored on the Greenland shark: growth of about
1 cm per year, sexual maturity at 156 +/- 22 years, 200-324
pups per pregnancy depending on maternal size, and a
radiocarbon lifespan estimate of 392 +/- 120 years.

The model is age-classed rather than stage-classed. Stage
classes with fixed-duration graduation blur a sharp maturity
age, and maturity age is one of the few things actually
measured for this species -- it should not be smeared for
notational convenience. Stage labels are retained as an
aggregation for reporting only.

Literature: Docs/literature.md sections 2, 7 and 9,
Docs/structure.md.
"""

import numpy as np

STAGES = ('pup', 'juvenile', 'subadult', 'adult')

# Age boundaries in years: pup 0-14, juvenile 15-74,
# subadult 75-149, adult 150+. The maturity boundary at 150 is
# empirical (156 +/- 22). The internal splits are a reporting
# convenience, not observed life-history boundaries.
STAGE_BOUNDS = (0, 15, 75, 150)

# Annual survival within each stage. Not measured -- no
# survival schedule exists for an animal that matures at 150
# years. This is the standard shape for a large slow
# elasmobranch: low early survival, high adult survival.
# Heuristic, and the most influential unmeasured input here.
STAGE_SURVIVAL = (0.85, 0.95, 0.98, 0.99)

MAX_AGE = 300               # working lifespan; 392 +/- 120 reported
MATURITY_AGE = 150          # 156 +/- 22 reported

# Litter size is empirical (200-324, midpoint used). Breeding
# interval is not known for this species; 10 years is a
# placeholder.
PUPS_PER_PREGNANCY = 262.0
BREEDING_INTERVAL_YEARS = 10.0
FEMALE_FRACTION = 0.5

RAW_FECUNDITY = (PUPS_PER_PREGNANCY / BREEDING_INTERVAL_YEARS
                 * FEMALE_FRACTION)


# ---------------------------------------------------------
# Life table and matrix construction
# ---------------------------------------------------------


def survival_schedule(max_age=MAX_AGE, bounds=STAGE_BOUNDS,
                      stage_survival=STAGE_SURVIVAL):
    """Annual survival for each age class, 0 .. max_age - 1."""
    ages = np.arange(max_age)
    idx = np.searchsorted(np.asarray(bounds[1:]), ages, side='right')
    return np.asarray(stage_survival, dtype=float)[idx]


def fecundity_schedule(fecundity, max_age=MAX_AGE,
                       maturity_age=MATURITY_AGE):
    """Female offspring per female per year, by age class."""
    ages = np.arange(max_age)
    return np.where(ages >= maturity_age, float(fecundity), 0.0)


def build_leslie_matrix(fecundity=RAW_FECUNDITY, max_age=MAX_AGE,
                        maturity_age=MATURITY_AGE,
                        bounds=STAGE_BOUNDS,
                        stage_survival=STAGE_SURVIVAL):
    """Age-classed projection matrix.

    Top row is fecundity by age; the subdiagonal is annual
    survival. Individuals reaching max_age leave the system,
    which is what bounds lifespan.
    """
    max_age = int(max_age)
    s = survival_schedule(max_age, bounds, stage_survival)
    m = fecundity_schedule(fecundity, max_age, maturity_age)

    A = np.zeros((max_age, max_age))
    A[0, :] = m
    A[1:, :-1] = np.diag(s[:-1])
    return A


def life_table(max_age=MAX_AGE, maturity_age=MATURITY_AGE,
               bounds=STAGE_BOUNDS, stage_survival=STAGE_SURVIVAL,
               fecundity=RAW_FECUNDITY):
    """Cumulative survivorship l(x) and fecundity m(x)."""
    max_age = int(max_age)
    s = survival_schedule(max_age, bounds, stage_survival)
    lx = np.concatenate(([1.0], np.cumprod(s)[:-1]))
    mx = fecundity_schedule(fecundity, max_age, maturity_age)
    return lx, mx


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------


def _sorted_eigen(A):
    values, vectors = np.linalg.eig(A)
    order = np.argsort(-np.abs(values))
    return values[order], vectors[:, order]


def growth_rate(A):
    """Dominant eigenvalue: annual multiplication rate."""
    values, _ = _sorted_eigen(A)
    return float(np.real(values[0]))


def stable_age_distribution(A):
    """Right eigenvector, normalised to sum to 1."""
    _, vectors = _sorted_eigen(A)
    w = np.abs(np.real(vectors[:, 0]))
    return w / w.sum()


def net_reproductive_rate(lx, mx):
    """R0: expected lifetime female offspring per female."""
    return float(np.sum(lx * mx))


def generation_time(lx, mx):
    """Mean age of the parents of a cohort, in years.

    Computed directly from the life table rather than as
    ln(R0)/ln(lambda), which is singular at lambda = 1 -- the
    exact case the baseline is calibrated to.

    This is the model's integration window. Compare it to
    baseline_lifespan_years: they are different quantities and
    warming moves them differently.
    """
    R0 = net_reproductive_rate(lx, mx)
    if R0 <= 0:
        return np.nan
    ages = np.arange(len(lx))
    return float(np.sum(ages * lx * mx) / R0)


def growth_rate_from_life_table(lx, mx, bounds=(1e-6, 1e6), tol=1e-12):
    """Solve Euler-Lotka for lambda: sum l(x) m(x) lambda^-x = 1.

    Equivalent to the dominant eigenvalue of the Leslie matrix but
    O(max_age) instead of O(max_age^3), which matters when sweeping
    a parameter grid. The left side decreases monotonically in
    lambda, so bisection is safe.
    """
    lx = np.asarray(lx, dtype=float)
    mx = np.asarray(mx, dtype=float)
    ages = np.arange(len(lx))
    weight = lx * mx

    if weight.sum() <= 0:
        return 0.0

    def residual(lam):
        return float(np.sum(weight * lam ** (-ages.astype(float)))) - 1.0

    lo, hi = bounds
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def damping_ratio(A):
    """Ratio of dominant to subdominant eigenvalue magnitude.

    The rate at which transient structure decays back to the
    stable age distribution. A ratio near 1 means the
    population converges slowly and carries the imprint of past
    conditions for a long time -- long ecological memory. A
    large ratio means it forgets quickly.

    This is the memory measure that should be used in place of
    lifespan.
    """
    values, _ = _sorted_eigen(A)
    if len(values) < 2 or abs(values[1]) < 1e-15:
        return np.inf
    return float(abs(values[0]) / abs(values[1]))


def memory_years(A, residual=0.1):
    """Years for a perturbation to decay to `residual` of its size.

    log(1/residual) / log(damping ratio), in years because the
    matrix is annual. The operational integration window.
    """
    rho = damping_ratio(A)
    if not np.isfinite(rho) or rho <= 1.0:
        return np.inf
    return float(np.log(1.0 / residual) / np.log(rho))


def transient_period(A):
    """Period of the damped oscillation following a perturbation.

    From the argument of the subdominant eigenvalue pair. For a
    long-lived species with a wide reproductive span this comes
    out near the generation time -- cohort echo. Returns inf if
    the subdominant eigenvalue is real (no oscillation).
    """
    values, _ = _sorted_eigen(A)
    if len(values) < 2:
        return np.inf
    theta = abs(np.angle(values[1]))
    if theta < 1e-12:
        return np.inf
    return float(2.0 * np.pi / theta)


def calibrate_fecundity(max_age=MAX_AGE, maturity_age=MATURITY_AGE,
                        bounds=STAGE_BOUNDS,
                        stage_survival=STAGE_SURVIVAL,
                        target_lambda=1.0):
    """Fecundity giving a chosen baseline growth rate.

    The baseline is calibrated to stationarity so forcing
    experiments read against a flat reference. A convention,
    not a claim that the real population is stationary -- it
    makes the effect of forcing legible by removing an
    arbitrary baseline trend.

    R0 is linear in fecundity, so at target_lambda = 1 this is
    exact rather than iterative.
    """
    lx, mx = life_table(max_age, maturity_age, bounds, stage_survival,
                        fecundity=1.0)
    if target_lambda == 1.0:
        R0_unit = net_reproductive_rate(lx, mx)
        return float('inf') if R0_unit <= 0 else 1.0 / R0_unit

    lo, hi = 1e-9, 1e9
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        A = build_leslie_matrix(mid, max_age, maturity_age, bounds,
                                stage_survival)
        if growth_rate(A) < target_lambda:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def aggregate_to_stages(n_by_age, bounds=STAGE_BOUNDS):
    """Collapse an age vector into the four reporting stages."""
    n_by_age = np.asarray(n_by_age, dtype=float)
    edges = list(bounds) + [n_by_age.shape[-1]]
    return np.stack([n_by_age[..., edges[i]:edges[i + 1]].sum(axis=-1)
                     for i in range(len(bounds))], axis=-1)


# ---------------------------------------------------------
# Forcing
# ---------------------------------------------------------


def warmed_ages(delta_T, Q10, max_age=MAX_AGE, maturity_age=MATURITY_AGE,
                bounds=STAGE_BOUNDS, decoupling=0.0):
    """Age axis compressed by metabolic acceleration.

    Warming raises rates, so ages are reached sooner: maturity
    arrives earlier and lifespan shortens together. `decoupling`
    damps the effect for the same reason
    maintenance_adjusted_lifespan() does -- longevity in this
    lineage involves active encoded maintenance, not only
    ambient chemistry.

    decoupling = 0 gives full rate-of-living compression,
    decoupling = 1 leaves the age axis untouched.

    Returns (max_age, maturity_age, bounds), all integers, each
    at least 1 so the matrix stays well formed.
    """
    multiplier = Q10 ** (delta_T / 10.0)
    effective = multiplier ** (1.0 - decoupling)
    scale = lambda x: max(1, int(round(x / effective)))

    new_bounds = tuple(sorted(set([0] + [scale(b) for b in bounds[1:]])))
    new_maturity = scale(maturity_age)
    new_max = max(new_maturity + 2, scale(max_age))
    return new_max, new_maturity, new_bounds


def allee_multiplier(adults, critical_density):
    """Recruitment penalty below a critical adult density.

    Implements additions.md item 4:
        if population < critical: recruitment *= population/critical

    Evidence in elasmobranchs is genuinely mixed. Mate
    limitation is the most commonly reported Allee mechanism in
    general, but detection of Allee effects in marine fishes
    carries known analytical biases, some depleted shark
    populations have recovered faster than demographic models
    predicted, and mating-system variation appears unlikely to
    be a major determinant of extinction vulnerability.

    Defaults to off (critical_density = 0) for that reason.
    Turning it on states a hypothesis; it does not correct an
    error.
    """
    if critical_density <= 0:
        return 1.0
    return float(min(1.0, max(0.0, adults / critical_density)))


def recruitment_transfer(A, forcing, maturity_age=MATURITY_AGE,
                         amplitude=0.1, discard=None):
    """How much recruitment variance reaches the adult population.

    Drives recruitment multiplicatively with `forcing` and returns
    (periods, gain), the ratio of the adult-abundance spectrum to
    the forcing spectrum, period by period.

    This is the test geometry.md's central claim was missing. The
    claim is that compressing ecological memory lets high-frequency
    environmental variation propagate further into the food web.
    Nothing in the model oscillated, so it could not be checked.
    Adults integrate every recruitment year from maturity to death,
    which makes them a low-pass filter whose cutoff is set by that
    span -- and warming shortens the span.

    Adult abundance is taken in logs and linearly detrended. With
    lambda at exactly 1 and multiplicative forcing, the population
    is a random walk in log space, so its variance grows without
    bound and a single variance ratio would be a function of
    however long the run happened to be. The spectrum is well
    behaved where the variance is not.
    """
    forcing = np.asarray(forcing, dtype=float)
    steps = len(forcing)
    discard = maturity_age * 4 if discard is None else int(discard)

    n0 = stable_age_distribution(A) * len(A)
    supply = 1.0 + amplitude * forcing
    traj = project(A, n0, steps, supply=supply, maturity_age=maturity_age)

    adults = traj[discard + 1:, maturity_age:].sum(axis=1)
    driver = forcing[discard:]
    if len(adults) < 16:
        raise ValueError('run too short after discarding the transient')

    response = np.log(np.maximum(adults, 1e-300))
    index = np.arange(len(response), dtype=float)
    response = response - np.polyval(np.polyfit(index, response, 1), index)

    window = np.hanning(len(response))
    response_fft = np.abs(np.fft.rfft(response * window))
    driver_fft = np.abs(np.fft.rfft((driver - driver.mean()) * window))
    freqs = np.fft.rfftfreq(len(response), d=1.0)

    keep = freqs > 0
    gain = response_fft[keep] / np.maximum(driver_fft[keep], 1e-12)
    return 1.0 / freqs[keep], gain


def band_gain(periods, gain, low, high):
    """Mean gain across a band of periods, e.g. the 2-7 year ENSO band."""
    periods = np.asarray(periods, dtype=float)
    inside = (periods >= low) & (periods <= high)
    return float(np.mean(np.asarray(gain)[inside])) if inside.any() else np.nan


def project(A, n0, steps, supply=None, critical_density=0.0,
            maturity_age=MATURITY_AGE):
    """Project the population forward, optionally forced.

    `supply` scales fecundity at each step -- the channel
    through which spatial supply and trophic energy reach
    demography. It is broadcast if scalar.

    Returns an array of shape (steps + 1, n_ages).
    """
    n0 = np.asarray(n0, dtype=float)
    supply_arr = (np.ones(steps) if supply is None
                  else np.broadcast_to(np.asarray(supply, dtype=float),
                                       (steps,)))

    out = np.empty((steps + 1, len(n0)))
    out[0] = n0
    state = n0.copy()
    top = A[0, :].copy()

    for k in range(steps):
        adults = state[maturity_age:].sum() if maturity_age < len(state) else 0.0
        scale = supply_arr[k] * allee_multiplier(adults, critical_density)
        recruits = float(top @ state) * scale
        state = np.concatenate(([recruits], A[1:, :-1].diagonal() * state[:-1]))
        out[k + 1] = state

    return out
